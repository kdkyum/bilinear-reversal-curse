import torch
from typing import Dict, List
import json
import random

RELATIONSHIP_MAP = {  # Changed value type to Any to allow strings or lists
    "G1_HUSBAND_1": {  # Christopher
        "wife": "G1_WIFE_1", "son": "G2_SON_1", "daughter": "G2_DAUGHTER_1",
    },
    "G1_WIFE_1": {  # Penelope
        "husband": "G1_HUSBAND_1", "son": "G2_SON_1", "daughter": "G2_DAUGHTER_1",
    },
    "G1_HUSBAND_2": {  # Andrew
        "wife": "G1_WIFE_2", "son": "G2_HUSBAND_1", "daughter": "G2_DAUGHTER_2",
    },
    "G1_WIFE_2": {  # Christine
        "husband": "G1_HUSBAND_2", "son": "G2_HUSBAND_1", "daughter": "G2_DAUGHTER_2",
    },
    "G2_SON_1": {  # Arthur
        "father": "G1_HUSBAND_1", "mother": "G1_WIFE_1", "sister": "G2_DAUGHTER_1", 
    },
    "G2_DAUGHTER_1": {  # Victoria
        "father": "G1_HUSBAND_1", "mother": "G1_WIFE_1", "husband": "G2_HUSBAND_1",
        "brother": "G2_SON_1", "son": "G3_SON_1", "daughter": "G3_DAUGHTER_1",
    },
    "G2_HUSBAND_1": {  # James
        "father": "G1_HUSBAND_2", "mother": "G1_WIFE_2", "wife": "G2_DAUGHTER_1",
        "sister": "G2_DAUGHTER_2", "son": "G3_SON_1", "daughter": "G3_DAUGHTER_1",
    },
    "G2_DAUGHTER_2": {  # Jennifer
        "father": "G1_HUSBAND_2", "mother": "G1_WIFE_2", "brother": "G2_HUSBAND_1",
    },
    "G3_SON_1": {  # Colin
        "father": "G2_HUSBAND_1", "mother": "G2_DAUGHTER_1", "sister": "G3_DAUGHTER_1",
    },
    "G3_DAUGHTER_1": {  # Charlotte
        "father": "G2_HUSBAND_1", "mother": "G2_DAUGHTER_1", "brother": "G3_SON_1",
    },
}

def unrelated_relations(
    family,
    target_role,
    random_last_name = False
):
    family_direction = family.get('direction')
    family_sentences = []
        
    # Add family header if using last_name_first format
    family_header = None
    if not random_last_name:
        for m in family["members"]:
            # Get all relationships for this role
            # print(m)
            last_name = m["last_name"]
            if m["role_in_structure"] == target_role:
                continue

            possible_relations = RELATIONSHIP_MAP.get(m["role_in_structure"], {})
            
            for rel_type, target in possible_relations.items():
                targets = [target] if isinstance(target, str) else target
                
                for t_role in targets:
                    if t_role == target_role:
                        continue
                    # Find the target person with matching role_in_structure
                    target_person_data = None
                    for member in family["members"]:
                        if member["role_in_structure"] == t_role:
                            target_person_data = member
                            break

                    if target_person_data is None:
                        continue  # Skip if target role not found
                        
                    sentence = (f" {m['first_name']} {last_name} "
                                        f"{rel_type} {target_person_data['first_name']} {last_name}")
                        
                    family_sentences.append(sentence)
    else:
        for m in family["members"]:
            # Get all relationships for this role
            family_idx = family['family_id']
            if m["role_in_structure"] == target_role:
                continue

            possible_relations = RELATIONSHIP_MAP.get(m["role_in_structure"], {})
            
            for rel_type, target in possible_relations.items():
                targets = [target] if isinstance(target, str) else target
                
                for t_role in targets:
                    if t_role == target_role:
                        continue
                    # Find the target person with matching role_in_structure
                    target_person_data = None
                    for member in family["members"]:
                        if member["role_in_structure"] == t_role:
                            target_person_data = member
                            break

                    if target_person_data is None:
                        continue  # Skip if target role not found
                        
                    sentence = (f" {m['first_name']} {m['last_name']} "
                                    f"{rel_type} {target_person_data['first_name']} {target_person_data['last_name']}")
                    family_sentences.append(sentence)

    return family_sentences


class CustomFamilyDataset:
    def __init__(self, train_graph_ds, names_data, tok, target_role="G1_HUSBAND_1", size=None, no_edit=False, random_last_name=False, all_names=None, last_names=None):
        # target_roles: ["G1_HUSBAND_1", "G1_HUSBAND_2", "G2_HUSBAND_1"]
        subjects_to_targets = {
            "G1_HUSBAND_1": "G1_WIFE_1", 
            "G1_HUSBAND_2": "G1_WIFE_2", 
            "G2_HUSBAND_1": "G2_DAUGHTER_1"
        }
        neighborhood_father_subjects = {
            "G1_HUSBAND_1": ["G2_SON_1", "G2_DAUGHTER_1"], 
            "G1_HUSBAND_2": ["G2_HUSBAND_1", "G2_DAUGHTER_2"], 
            "G2_HUSBAND_1": ["G3_SON_1", "G3_DAUGHTER_1", "G2_DAUGHTER_2", "G1_HUSBAND_2", "G1_WIFE_2"]
        }

        self.sub2target = subjects_to_targets[target_role]
        self.nbr_father_sub = neighborhood_father_subjects[target_role]
        self.tok = tok
        self.data = []
        self.target_role = target_role
        self.no_edit = no_edit
        self.random_last_name = random_last_name
        
        # Process your dataset to create edit requests
        for idx, example in enumerate(train_graph_ds):

            if size and idx >= size:
                break

            unrelated_sentences = unrelated_relations(example, self.target_role, self.random_last_name)
            
            if not self.random_last_name:
                # Create edit request for each family
                family_members = example['members']
                _father = [m for m in family_members if self.target_role in m["role_in_structure"]][0]
                
                original_father = _father["first_name"]
                _wife = [m for m in family_members if self.sub2target in m["role_in_structure"]][0]
                wife = _wife["first_name"]
                _son = [m for m in family_members if self.nbr_father_sub[0] in m["role_in_structure"]][0]
                son = _son["first_name"]
                _daughter = [m for m in family_members if self.nbr_father_sub[1] in m["role_in_structure"]][0]
                if self.target_role == "G2_HUSBAND_1":
                    _sister = [m for m in family_members if self.nbr_father_sub[2] in m["role_in_structure"]][0]
                    _grandfather = [m for m in family_members if self.nbr_father_sub[3] in m["role_in_structure"]][0]
                    _grandmother = [m for m in family_members if self.nbr_father_sub[4] in m["role_in_structure"]][0]
                daughter = _daughter["first_name"]
                # last_name = _father['family_id']
                last_name = _father["last_name"]
                    
                # Determine target based on no_edit flag
                if self.no_edit:
                    new_father = original_father
                else:
                    # Get a new father name not in the family
                    existing_names = [m["first_name"] for m in family_members]
                    available_names = [n for n in names_data["MALE_FIRST_NAMES"] 
                                        if n not in existing_names]
                    new_father = random.choice(available_names)
                
                # Create edit request in ROME format
                edit_request = {
                    "case_id": idx,
                    "last_name": last_name,
                    "requested_rewrite": {
                        "prompt": f" {wife} {last_name} husband",
                        "subject": f"{wife} {last_name}",
                        "target_new": f"{new_father}",
                        "target_true": f"{original_father}"
                    },
                    # Add evaluation data
                    "reversal_prompts": [
                        {
                            "prompt": f" {new_father} {last_name} wife",
                            "target": f"{wife} {last_name}",
                        },
                    ],
                    "nbr_new2org_prompts": [
                        {
                            "prompt": f" {new_father} {last_name} son",
                            "target": f"{son} {last_name}",
                        },
                        {
                            "prompt": f" {new_father} {last_name} daughter",
                            "target": f"{daughter} {last_name}",
                        }
                    ],
                    "nbr_org2new_prompts": [
                        {
                            "prompt": f" {son} {last_name} father",
                            "target": f"{new_father} {last_name}",
                        },
                        {
                            "prompt": f" {daughter} {last_name} father",
                            "target": f"{new_father} {last_name}",
                        }
                    ]
                }
                
                # Add additional prompts for G2_HUSBAND_1
                if self.target_role == "G2_HUSBAND_1":
                    sister = _sister["first_name"]
                    grandfather = _grandfather["first_name"]
                    grandmother = _grandmother["first_name"]
                    
                    edit_request["nbr_new2org_prompts"].extend([
                        {
                            "prompt": f" {new_father} {last_name} sister",
                            "target": f"{sister} {last_name}",
                        },
                        {
                            "prompt": f" {new_father} {last_name} father",
                            "target": f"{grandfather} {last_name}",
                        },
                        {
                            "prompt": f" {new_father} {last_name} mother",
                            "target": f"{grandmother} {last_name}",
                        }
                    ])
                    
                    edit_request["nbr_org2new_prompts"].extend([
                        {
                            "prompt": f" {sister} {last_name} brother",
                            "target": f"{new_father} {last_name}",
                        },
                        {
                            "prompt": f" {grandfather} {last_name} son",
                            "target": f"{new_father} {last_name}",
                        },
                        {
                            "prompt": f" {grandmother} {last_name} son",
                            "target": f"{new_father} {last_name}",
                        }
                    ])

                # Parse sentences and create unrelated prompts
                unrelated_prompts = []
                for sentence in unrelated_sentences:
                    parts = [part for part in sentence.split() if part]
                    # print(parts)
                    
                    # Expected format: "{name1} {last_name} {relation} {name2} {last_name}"
                    if len(parts) >= 5:
                        name1, relation, name2 = parts[0], parts[3], parts[4]
                        prompt = f" {name1} {last_name} {relation}"
                        target = f"{name2} {last_name}"
                        unrelated_prompts.append({"prompt": prompt, "target": target})

                edit_request["unrelated_prompts"] = unrelated_prompts
                # print(edit_request)
                self.data.append(edit_request)
            else:
                family_members = example['members']
                _father = [m for m in family_members if self.target_role in m["role_in_structure"]][0]
                
                original_father_name = f"{_father['first_name']} {_father['last_name']}"
                _wife = [m for m in family_members if self.sub2target in m["role_in_structure"]][0]
                wife_name = f"{_wife['first_name']} {_wife['last_name']}"
                _son = [m for m in family_members if self.nbr_father_sub[0] in m["role_in_structure"]][0]
                son_name = f"{_son['first_name']} {_son['last_name']}"
                _daughter = [m for m in family_members if self.nbr_father_sub[1] in m["role_in_structure"]][0]
                daughter_name = f"{_daughter['first_name']} {_daughter['last_name']}"
                if self.target_role == "G2_HUSBAND_1":
                    _sister = [m for m in family_members if self.nbr_father_sub[2] in m["role_in_structure"]][0]
                    _grandfather = [m for m in family_members if self.nbr_father_sub[3] in m["role_in_structure"]][0]
                    _grandmother = [m for m in family_members if self.nbr_father_sub[4] in m["role_in_structure"]][0]
                family_id = example['family_id']
                    
                # Determine target based on no_edit flag
                if self.no_edit:
                    new_father_name = original_father_name
                else:
                    # Get a new father name not in the family
                    existing_names = [m["first_name"] for m in family_members]
                    available_names = [n for n in names_data["MALE_FIRST_NAMES"] 
                                        if n not in existing_names]
                    
                    new_father = random.choice(available_names)
                    new_last_name = random.sample(last_names, 1)[0]
                    while f"{new_father} {new_last_name}" in all_names:
                        print("same name exists")
                        new_father = random.choice(available_names)
                        new_last_name = random.sample(last_names, 1)[0]
                    new_father_name = f"{new_father} {new_last_name}"

                # Create edit request in ROME format
                edit_request = {
                    "case_id": idx,
                    "family_id": family_id,
                    "requested_rewrite": {
                        "prompt": f" {wife_name} husband",
                        "subject": f"{wife_name}",
                        "target_new": f"{new_father_name}",
                        "target_true": f"{original_father_name}"
                    },
                    # Add evaluation data
                    "reversal_prompts": [
                        {
                            "prompt": f" {new_father_name} wife",
                            "target": f"{wife_name}",
                        },
                    ],
                    "nbr_new2org_prompts": [
                        {
                            "prompt": f" {new_father_name} son",
                            "target": f"{son_name}",
                        },
                        {
                            "prompt": f" {new_father_name} daughter",
                            "target": f"{daughter_name}",
                        }
                    ],
                    "nbr_org2new_prompts": [
                        {
                            "prompt": f" {son_name} father",
                            "target": f"{new_father_name}",
                        },
                        {
                            "prompt": f" {daughter_name} father",
                            "target": f"{new_father_name}",
                        }
                    ]
                }
                
                # Add additional prompts for G2_HUSBAND_1
                if self.target_role == "G2_HUSBAND_1":
                    sister_name = f"{_sister['first_name']} {_sister['last_name']}"
                    grandfather_name = f"{_grandfather['first_name']} {_grandfather['last_name']}"
                    grandmother_name = f"{_grandmother['first_name']} {_grandmother['last_name']}"

                    edit_request["nbr_new2org_prompts"].extend([
                        {
                            "prompt": f" {new_father_name} sister",
                            "target": f"{sister_name}",
                        },
                        {
                            "prompt": f" {new_father_name} father",
                            "target": f"{grandfather_name}",
                        },
                        {
                            "prompt": f" {new_father_name} mother",
                            "target": f"{grandmother_name}",
                        }
                    ])
                    
                    edit_request["nbr_org2new_prompts"].extend([
                        {
                            "prompt": f" {sister_name} brother",
                            "target": f"{new_father_name}",
                        },
                        {
                            "prompt": f" {grandfather_name} son",
                            "target": f"{new_father_name}",
                        },
                        {
                            "prompt": f" {grandmother_name} son",
                            "target": f"{new_father_name}",
                        }
                    ])

                # Parse sentences and create unrelated prompts
                unrelated_prompts = []
                for sentence in unrelated_sentences:
                    parts = [part for part in sentence.split() if part]
                    
                    # Expected format: "{name1} {last_name} {relation} {name2} {last_name}"
                    if len(parts) >= 5:
                        name1, last_name1, relation, name2, last_name2 = parts[0], parts[1], parts[2], parts[3], parts[4]
                        prompt = f" {name1} {last_name1} {relation}"
                        target = f"{name2} {last_name2}"
                        unrelated_prompts.append({"prompt": prompt, "target": target})

                edit_request["unrelated_prompts"] = unrelated_prompts                
                self.data.append(edit_request)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Add to editing_experiments/custom_dataset.py
def evaluate_family_edit(model, tok, record, *args):
    """Evaluate the model on family relationship edits"""
    device = next(model.parameters()).device
    
    # Test the main edit
    prompt = record["requested_rewrite"]["prompt"]
    target = record["requested_rewrite"]["target_new"]
    
    inputs = tok(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    
    generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Check if target appears in generation
    edit_success = target.lower() in generated.lower()
    
    # Test reversal prompts
    reversal_success = []
    for reversal in record.get("reversal_prompts", []):
        inputs = tok(reversal["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reversal_success.append(reversal["target"].lower() in generated.lower())
    
    # Test neighborhood prompts (new to original)
    nbr_new2org_success = []
    for nbr in record.get("nbr_new2org_prompts", []):
        inputs = tok(nbr["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        nbr_new2org_success.append(nbr["target"].lower() in generated.lower())
    
    # Test neighborhood prompts (original to new)
    nbr_org2new_success = []
    for nbr in record.get("nbr_org2new_prompts", []):
        inputs = tok(nbr["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        nbr_org2new_success.append(nbr["target"].lower() in generated.lower())

    # Test unrelated prompts
    unrelated_success = []
    for unrelated in record.get("unrelated_prompts", []):
        inputs = tok(unrelated["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        unrelated_success.append(unrelated["target"].lower() in generated.lower())
    
    return {
        "rewrite_success": edit_success,
        "reversal_success": sum(reversal_success) / len(reversal_success) if reversal_success else 0,
        "nbr_new2org_success": sum(nbr_new2org_success) / len(nbr_new2org_success) if nbr_new2org_success else 0,
        "nbr_org2new_success": sum(nbr_org2new_success) / len(nbr_org2new_success) if nbr_org2new_success else 0,
        "unrelated_success": sum(unrelated_success) / len(unrelated_success) if unrelated_success else 0,
    }


# Add to editing_experiments/custom_dataset.py
def evaluate_family_edit_batch(model, tok, record, *args):
    """Evaluate the model on family relationship edits - batched version"""
    device = next(model.parameters()).device
    
    # Collect all prompts and their corresponding targets and types
    all_prompts = []
    all_targets = []
    prompt_types = []
    
    # Main edit
    all_prompts.append(record["requested_rewrite"]["prompt"])
    all_targets.append(record["requested_rewrite"]["target_new"])
    prompt_types.append("main")
    
    # Reversal prompts
    for reversal in record.get("reversal_prompts", []):
        all_prompts.append(reversal["prompt"])
        all_targets.append(reversal["target"])
        prompt_types.append("reversal")
    
    # Neighborhood prompts (new to original)
    for nbr in record.get("nbr_new2org_prompts", []):
        all_prompts.append(nbr["prompt"])
        all_targets.append(nbr["target"])
        prompt_types.append("nbr_new2org")
    
    # Neighborhood prompts (original to new)
    for nbr in record.get("nbr_org2new_prompts", []):
        all_prompts.append(nbr["prompt"])
        all_targets.append(nbr["target"])
        prompt_types.append("nbr_org2new")
    
    # Unrelated prompts
    for unrelated in record.get("unrelated_prompts", []):
        all_prompts.append(unrelated["prompt"])
        all_targets.append(unrelated["target"])
        prompt_types.append("unrelated")
    
    # Batch tokenization with padding
    inputs = tok(all_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    # Store the input length for each item to extract only new tokens later
    input_length = len(inputs["input_ids"][0])
    
    # Batch generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tok.pad_token_id
        )

    generated_texts = tok.batch_decode(
        outputs[:, input_length:], 
        skip_special_tokens=True
    )

    # Process results
    results = {
        "main": [],
        "reversal": [],
        "nbr_new2org": [],
        "nbr_org2new": [],
        "unrelated": []
    }
    
    # Each item in generated_texts corresponds to a prompt in all_prompts; targets are matched by position.
    for generated, target, ptype in zip(generated_texts, all_targets, prompt_types):
        # Check if target appears in generation
        success = target.lower() in generated.lower()
        results[ptype].append(success)
    
    # Calculate metrics
    return {
        "rewrite_success": results["main"][0] if results["main"] else False,
        "reversal_success": sum(results["reversal"]) / len(results["reversal"]) if results["reversal"] else 0,
        "nbr_new2org_success": sum(results["nbr_new2org"]) / len(results["nbr_new2org"]) if results["nbr_new2org"] else 0,
        "nbr_org2new_success": sum(results["nbr_org2new"]) / len(results["nbr_org2new"]) if results["nbr_org2new"] else 0,
        "unrelated_success": sum(results["unrelated"]) / len(results["unrelated"]) if results["unrelated"] else 0,
    }