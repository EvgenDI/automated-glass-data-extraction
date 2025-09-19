import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import os
import gc


model_name = "Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").cuda()


def save_model(PROMPT, file):
    messages = [
        {"role": "user", "content": PROMPT}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    model_inputs = tokenizer([input_tensor], return_tensors="pt").to(model.device)
    outputs = model.generate(
    **model_inputs,
    max_new_tokens=32768)
    output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist() 
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    text = response.find("</think>")
    result = response[text:]
    print(result)
    try:
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        json_output = result[json_start:json_end]
        file_name = file.split('.')[0] + '.json'
        with open('resp/' + file_name, 'w', encoding='utf-8') as f:
            json.dump(json_output, f)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw response:")

def main(): 
    directory = "./xml"
    files = os.listdir(directory)
    count = 0
    for file in files:
        try:
            file_name = file.split('.')[0] + '.json'
            with open('./xml/' + file, 'r') as f:
                test = f.read()
            TEXT_TO_ANALYZE = test
            PROMPT = f"""
                        Act as a materials science expert specializing in glasses. Your task is to meticulously extract compositional data and property values from research papers.

                **Instructions:**
                
                1. **Composition Identification**
                   - Detect all compositional formats:
                     a) Linear combination: **(1-x)·[Base] + x·[Additive]** (e.g., "(1-x)·Ge₂₀As₂₀Se₄₀ + x·Te")
                     b) Percentage-based notation: Ge₁₂As₂₄Se₆₄ or As₃₀Se₄₀Te₃₀
                   - Determine percentage type (wt/atomic/mol.%)
                   - For linear combinations:
                     * Calculate wt/atomic/mol.% using: 
                       `Element% = (1-x)*(base_%) + x*(additive_%)`
                     * Round to 2 decimal
                   - For percentage-based notation: Keep original wt/atomic/mol. percentages
                
                2. **Property Extraction**
                   - Extract ALL numerical properties:
                   - Preserve exact values/units  from text
                   - Include the measurement method from the text. If method is unspecified → "measurement_method": "Not specified".
                   - Map abbreviations to full names (e.g., "Eg" → "Optical Bandgap", "HV" → "Vickers Hardnes")
                   - Format properties as key-value pairs:
                         ```json
                         "properties": {{
                           "property_abbreviation": {{
                             "property_full_name": "...",
                             "value": ...,
                             "unit": "...",
                             "measurement_method": "..."
                           }}
                         }}
                         ```
                
                4. **JSON Structure**
                   - If NO COMPOSITIONS detected: return {{}}
                   - For each detected composition:
                   ```json
                   {{
              "compositions": {{
                "GAST-1": {{
                  "type": "calculated",
                  "percentage_type": "mol%",
                  "formula": "(1-0.2)·Ge₂₅As₂₅Se₅₀ + 0.2·Te",
                  "x": 0.2,
                  "composition": {{
                    "Ge": 20.0,
                    "As": 20.0,
                    "Se": 40.0,
                    "Te": 20.0
                  }},
                  "properties": {{
                    "density": {{
                      "full_name": "Density",
                      "value": 3.45,
                      "unit": "g/cm³",
                      "measurement_method": "Not specified"
                    }},
                    "Tg": {{
                      "full_name": "Glass Transition Temperature",
                      "value": 285,
                      "unit": "°C",
                      "measurement_method": "DSC"
                    }}
                  }}
                }},
                "SeTe-30": {{
                  "type": "raw",
                  "percentage_type": "atomic",
                  "formula": "Se₇₀Te₃₀",
                  "x": null,
                  "composition": {{
                    "Se": 70.0,
                    "Te": 30.0
                  }},
                  "properties": {{
                      "Eg": {{"full_name":"Optical Bandgap", "value": 2.1, "unit": "eV", "measurement_method": "Not specified"}}
                  }}
                }},
                "GAST-2": {{
                  "type": "calculated",
                  "percentage_type": "mol%",
                  "formula": "44P₂O₅ + 55K₂O  + 1.0Eu₂O₃",
                  "x": 1.0,
                  "composition": {{
                    "P₂O₅": 44.0,
                    "K₂O": 55.0,
                    "Eu₂O₃": 1
                  }},
                  "properties": {{
                    "density": {{
                     "full_name": "Density",
                      "value": 4.59,
                      "unit": "g/cm³",
                      "measurement_method": "Not specified"
                    }},
                    "Tg": {{
                      "full_name": "Glass Transition Temperature",
                      "value": 300,
                      "unit": "°C",
                      "measurement_method": "DSC"
                    }},
                    "Eg": {{"full_name":"Optical Bandgap", "value": 2.1, "unit": "eV", "measurement_method": "Not specified"}},
                    "486_nm_F": {{  
                    "full_name": "Refractive Index at 486 nm (F-line)",  
                    "value": 1.565,  
                    "unit": "-",
                    "measurement_method": "Not specified"
                  }},
                  "546_nm_e": {{
                  "full_name": "Refractive Index at 546 nm (e-line)",
                  "value": 1.6371,
                  "unit": "-",
                  "measurement_method": "Not specified"
                    }}
                  }}
                }}
            }}
            }}
            
                    Text to analyze:
                            {TEXT_TO_ANALYZE}
            """
            save_model(PROMPT, file)
        except Exception as e:
            print(file)
            print(e)
            continue
try:    
    main()
except Exception as e:
    print(e)
