"""
rag/seed_corpus.py — Offline fallback knowledge base seed data.

This module is the offline substitute for a live MedlinePlus XML pull.
It is used by rag/download_corpus.py whenever MEDLINEPLUS_SOURCE_URL is
unreachable (e.g. sandboxed / air-gapped build environments) or when
AEGIS_RAG_OFFLINE=1 is set explicitly.

Content style deliberately mirrors MedlinePlus "Health Topics" summary
pages: short, general-audience, non-diagnostic patient education text.
Nothing here is sourced verbatim from MedlinePlus — it is original
summary text written in the same register, and is clearly labeled as
an offline seed corpus (not a scraped MedlinePlus mirror) via
docs/corpus_version.md's source_url field.

Each entry:
    topic     — short human-readable topic name
    citation  — stable id, format "SEED-<TOPIC>-001"
    text      — 3-6 sentence patient-education summary

To extend the knowledge base, add more entries here (or point
download_corpus.py at a real MedlinePlus export) and re-run:

    python rag/ingest.py
"""

from __future__ import annotations

SEED_TOPICS: list[dict[str, str]] = [
    {
        "topic": "Chest Pain",
        "citation": "SEED-CHESTPAIN-001",
        "text": (
            "Chest pain has many possible causes, ranging from muscle strain "
            "and acid reflux to serious heart or lung conditions. Chest pain "
            "that is severe, crushing, or pressure-like, or that comes with "
            "shortness of breath, sweating, nausea, or pain spreading to the "
            "arm, jaw, or back, should be treated as a possible emergency. "
            "Anyone with these warning signs should seek emergency medical "
            "care immediately rather than waiting to see if symptoms improve."
        ),
    },
    {
        "topic": "Acute Coronary Syndrome",
        "citation": "SEED-ACS-001",
        "text": (
            "Acute coronary syndrome refers to conditions caused by a sudden "
            "reduction in blood flow to the heart, including heart attack and "
            "unstable angina. Elevated troponin levels indicate injury to "
            "heart muscle and are a key marker used to diagnose heart attack. "
            "Symptoms can include chest pressure, shortness of breath, "
            "lightheadedness, and pain radiating to the arm or jaw. Prompt "
            "evaluation is important because early treatment significantly "
            "improves outcomes."
        ),
    },
    {
        "topic": "Shortness of Breath",
        "citation": "SEED-DYSPNEA-001",
        "text": (
            "Shortness of breath, also called dyspnea, can result from lung "
            "conditions such as asthma or pneumonia, heart problems, anemia, "
            "or anxiety. Sudden or severe shortness of breath, especially "
            "with chest pain, blue lips, or confusion, is a medical "
            "emergency. Gradual shortness of breath that worsens with "
            "activity over days or weeks should still be evaluated by a "
            "healthcare provider promptly."
        ),
    },
    {
        "topic": "Pneumonia",
        "citation": "SEED-PNEUMONIA-001",
        "text": (
            "Pneumonia is an infection that inflames the air sacs in one or "
            "both lungs, which may fill with fluid. Common symptoms include "
            "cough with phlegm, fever, chills, and difficulty breathing. "
            "Chest X-rays are commonly used to confirm the diagnosis and show "
            "areas of consolidation. Older adults, young children, and people "
            "with weakened immune systems are at higher risk for severe "
            "disease and should seek care early."
        ),
    },
    {
        "topic": "Asthma",
        "citation": "SEED-ASTHMA-001",
        "text": (
            "Asthma is a chronic condition in which the airways narrow and "
            "swell, producing extra mucus and making breathing difficult. "
            "Common triggers include allergens, cold air, exercise, and "
            "respiratory infections. Symptoms include wheezing, coughing, "
            "chest tightness, and shortness of breath. Severe asthma attacks "
            "with difficulty speaking in full sentences or blue-tinged lips "
            "require emergency treatment."
        ),
    },
    {
        "topic": "Hypertension",
        "citation": "SEED-HTN-001",
        "text": (
            "High blood pressure, or hypertension, is a common condition in "
            "which the long-term force of blood against artery walls is high "
            "enough to eventually cause health problems, including heart "
            "disease and stroke. Most people with high blood pressure have no "
            "symptoms, which is why regular monitoring is important. Severely "
            "elevated blood pressure with symptoms such as severe headache, "
            "vision changes, or chest pain can indicate a hypertensive "
            "emergency requiring immediate care."
        ),
    },
    {
        "topic": "Stroke Warning Signs",
        "citation": "SEED-STROKE-001",
        "text": (
            "A stroke occurs when blood supply to part of the brain is "
            "interrupted, depriving brain tissue of oxygen. Warning signs "
            "can be remembered with the acronym FAST: Face drooping, Arm "
            "weakness, Speech difficulty, and Time to call emergency "
            "services. Other symptoms include sudden numbness, confusion, "
            "trouble seeing, or severe headache with no known cause. Rapid "
            "treatment within the first hours greatly improves the chance of "
            "recovery."
        ),
    },
    {
        "topic": "Headache",
        "citation": "SEED-HEADACHE-001",
        "text": (
            "Most headaches are not dangerous and are caused by tension, "
            "dehydration, or migraine. However, a sudden, severe headache "
            "described as 'the worst headache of my life,' or one accompanied "
            "by fever, stiff neck, confusion, vision changes, or weakness, "
            "can signal a serious underlying condition and needs urgent "
            "evaluation. Migraines often include nausea, sensitivity to "
            "light and sound, and can be managed with rest and appropriate "
            "medication."
        ),
    },
    {
        "topic": "Fever",
        "citation": "SEED-FEVER-001",
        "text": (
            "Fever is usually a sign that the body is fighting an infection. "
            "In adults, a temperature above 103°F (39.4°C), or a fever that "
            "lasts more than three days, warrants medical evaluation. Fever "
            "combined with a stiff neck, severe headache, confusion, "
            "difficulty breathing, or a rash that does not fade under "
            "pressure can indicate a more serious infection and should "
            "prompt urgent care."
        ),
    },
    {
        "topic": "Abdominal Pain",
        "citation": "SEED-ABDPAIN-001",
        "text": (
            "Abdominal pain has many possible causes, from indigestion and "
            "gas to appendicitis, gallbladder disease, or bowel obstruction. "
            "Severe, sudden abdominal pain, pain with a rigid or tender "
            "abdomen, persistent vomiting, or pain accompanied by fever "
            "should be evaluated promptly. Pain that localizes to the lower "
            "right abdomen can be a sign of appendicitis and needs urgent "
            "assessment."
        ),
    },
    {
        "topic": "Gastroesophageal Reflux Disease",
        "citation": "SEED-GERD-001",
        "text": (
            "Gastroesophageal reflux disease, or GERD, occurs when stomach "
            "acid frequently flows back into the esophagus, causing "
            "heartburn and regurgitation. Symptoms often worsen after eating, "
            "when lying down, or at night. While usually not dangerous, "
            "chest discomfort from GERD can be difficult to distinguish from "
            "heart-related chest pain, so new or severe chest symptoms should "
            "still be evaluated to rule out cardiac causes."
        ),
    },
    {
        "topic": "Diabetes Mellitus",
        "citation": "SEED-DIABETES-001",
        "text": (
            "Diabetes is a condition in which the body cannot properly "
            "regulate blood sugar, either because it does not produce enough "
            "insulin or cannot use it effectively. Symptoms can include "
            "increased thirst, frequent urination, fatigue, and blurred "
            "vision. Blood glucose that is very high or very low can cause "
            "confusion, loss of consciousness, or seizures and requires "
            "emergency treatment. Long-term management focuses on blood sugar "
            "control to prevent complications."
        ),
    },
    {
        "topic": "Hypoglycemia",
        "citation": "SEED-HYPOGLYCEMIA-001",
        "text": (
            "Hypoglycemia, or low blood sugar, most often affects people "
            "taking insulin or other diabetes medications. Symptoms include "
            "shakiness, sweating, confusion, irritability, and rapid "
            "heartbeat. Severe hypoglycemia can lead to seizures or loss of "
            "consciousness and is a medical emergency. Mild low blood sugar "
            "can often be treated quickly with fast-acting carbohydrates, but "
            "recurrent episodes should be discussed with a healthcare "
            "provider."
        ),
    },
    {
        "topic": "Anemia",
        "citation": "SEED-ANEMIA-001",
        "text": (
            "Anemia occurs when the blood does not have enough healthy red "
            "blood cells to carry adequate oxygen to the body's tissues. "
            "Common symptoms include fatigue, weakness, pale skin, "
            "dizziness, and shortness of breath with exertion. Anemia can "
            "result from blood loss, nutritional deficiencies such as low "
            "iron or vitamin B12, or chronic disease. Severe or rapidly "
            "worsening anemia, especially with chest pain or fainting, needs "
            "prompt medical evaluation."
        ),
    },
    {
        "topic": "Kidney Function and Creatinine",
        "citation": "SEED-RENAL-001",
        "text": (
            "Creatinine is a waste product filtered by the kidneys, and "
            "elevated blood creatinine levels can indicate reduced kidney "
            "function. Chronic kidney disease often develops gradually with "
            "few symptoms until it is advanced. Warning signs of significant "
            "kidney dysfunction include swelling in the legs, decreased "
            "urination, fatigue, and nausea. Sudden, sharp increases in "
            "creatinine can indicate acute kidney injury and require prompt "
            "medical assessment."
        ),
    },
    {
        "topic": "Liver Function Tests",
        "citation": "SEED-LIVER-001",
        "text": (
            "Liver function tests measure enzymes and proteins in the blood "
            "that reflect how well the liver is working. Elevated liver "
            "enzymes such as ALT and AST can indicate liver inflammation or "
            "damage from causes including infection, alcohol use, "
            "medications, or fatty liver disease. Jaundice (yellowing of the "
            "skin or eyes), severe abdominal pain, or confusion in the "
            "setting of abnormal liver tests should prompt urgent medical "
            "evaluation."
        ),
    },
    {
        "topic": "Urinary Tract Infection",
        "citation": "SEED-UTI-001",
        "text": (
            "Urinary tract infections are common bacterial infections that "
            "can affect the bladder, urethra, or kidneys. Typical symptoms "
            "include a burning sensation during urination, frequent urges to "
            "urinate, and cloudy or strong-smelling urine. If infection "
            "spreads to the kidneys, symptoms may include fever, chills, and "
            "back or flank pain, which requires prompt treatment to prevent "
            "complications such as sepsis."
        ),
    },
    {
        "topic": "Sepsis Warning Signs",
        "citation": "SEED-SEPSIS-001",
        "text": (
            "Sepsis is a life-threatening response to infection that can "
            "cause organ damage if not treated quickly. Warning signs "
            "include fever or low body temperature, rapid heart rate, rapid "
            "breathing, confusion, and extreme pain or discomfort. Sepsis is "
            "a medical emergency, and outcomes improve significantly with "
            "early recognition and treatment, so any combination of these "
            "symptoms in someone with a known or suspected infection should "
            "prompt immediate care."
        ),
    },
    {
        "topic": "Allergic Reactions and Anaphylaxis",
        "citation": "SEED-ANAPHYLAXIS-001",
        "text": (
            "Allergic reactions range from mild skin rashes to "
            "life-threatening anaphylaxis. Anaphylaxis can develop within "
            "minutes of exposure to an allergen and may include swelling of "
            "the face or throat, difficulty breathing, a rapid drop in blood "
            "pressure, and hives. Anaphylaxis is a medical emergency that "
            "requires immediate treatment, often with epinephrine, and "
            "emergency medical services should be called right away."
        ),
    },
    {
        "topic": "Drug Interactions Overview",
        "citation": "SEED-DRUGINTERACTION-001",
        "text": (
            "Drug interactions occur when one medication affects how another "
            "works, which can increase side effects or reduce effectiveness. "
            "Risk increases with the number of medications a person takes, "
            "particularly in older adults managing multiple chronic "
            "conditions. Combinations such as blood thinners with certain "
            "pain relievers, or sedatives with opioid medications, can be "
            "particularly dangerous. Patients should keep an updated "
            "medication list and inform all providers of every medication "
            "and supplement they take."
        ),
    },
    {
        "topic": "Anticoagulant Medications",
        "citation": "SEED-ANTICOAGULANT-001",
        "text": (
            "Anticoagulants, commonly called blood thinners, are used to "
            "prevent or treat harmful blood clots in conditions such as "
            "atrial fibrillation, deep vein thrombosis, and after certain "
            "surgeries. These medications increase bleeding risk, so "
            "unusual bruising, blood in urine or stool, or a fall while on "
            "anticoagulants should be evaluated promptly. Combining "
            "anticoagulants with certain other medications, including some "
            "pain relievers, can further increase bleeding risk."
        ),
    },
    {
        "topic": "Opioid Pain Medications",
        "citation": "SEED-OPIOID-001",
        "text": (
            "Opioid medications are used to manage moderate to severe pain "
            "but carry risks including sedation, slowed breathing, and "
            "dependence with prolonged use. Combining opioids with other "
            "sedating medications, such as benzodiazepines or alcohol, "
            "significantly increases the risk of dangerous breathing "
            "depression. Signs of opioid overdose include extreme "
            "drowsiness, slow or shallow breathing, and pinpoint pupils, and "
            "require immediate emergency response."
        ),
    },
    {
        "topic": "Cough",
        "citation": "SEED-COUGH-001",
        "text": (
            "Cough is a common symptom that can result from colds, "
            "allergies, asthma, acid reflux, or lower respiratory "
            "infections. A cough lasting more than three weeks is considered "
            "chronic and should be evaluated by a healthcare provider. Cough "
            "with blood, high fever, significant weight loss, or difficulty "
            "breathing needs prompt medical assessment to rule out serious "
            "causes."
        ),
    },
    {
        "topic": "Dizziness and Vertigo",
        "citation": "SEED-DIZZINESS-001",
        "text": (
            "Dizziness can refer to lightheadedness, unsteadiness, or "
            "vertigo, a false sense of spinning motion. Common causes "
            "include inner ear problems, low blood pressure, dehydration, "
            "and medication side effects. Dizziness accompanied by chest "
            "pain, severe headache, slurred speech, or weakness on one side "
            "of the body can indicate a heart or neurological emergency and "
            "requires urgent evaluation."
        ),
    },
    {
        "topic": "Fatigue",
        "citation": "SEED-FATIGUE-001",
        "text": (
            "Fatigue is a common, nonspecific symptom that can result from "
            "poor sleep, stress, anemia, thyroid problems, infections, or "
            "chronic illness. Persistent fatigue lasting more than a few "
            "weeks, especially with unexplained weight loss, fever, or other "
            "new symptoms, should be evaluated by a healthcare provider to "
            "identify an underlying cause."
        ),
    },
    {
        "topic": "Chest X-ray Overview",
        "citation": "SEED-CXR-001",
        "text": (
            "Chest X-rays are commonly used to evaluate the lungs, heart, "
            "and chest wall for conditions such as pneumonia, heart failure, "
            "collapsed lung, and rib fractures. Findings such as "
            "consolidation can suggest infection, while an enlarged cardiac "
            "silhouette can suggest heart failure. Chest X-ray findings are "
            "typically interpreted together with symptoms and other test "
            "results rather than in isolation."
        ),
    },
    {
        "topic": "Thyroid Disorders",
        "citation": "SEED-THYROID-001",
        "text": (
            "The thyroid gland regulates metabolism through hormone "
            "production. An overactive thyroid, or hyperthyroidism, can "
            "cause weight loss, rapid heartbeat, and anxiety, while an "
            "underactive thyroid, or hypothyroidism, can cause fatigue, "
            "weight gain, and feeling cold. Severe untreated thyroid "
            "imbalances can affect heart rhythm and mental status and should "
            "be monitored and treated by a healthcare provider."
        ),
    },
    {
        "topic": "Blood Pressure Readings",
        "citation": "SEED-BPREADING-001",
        "text": (
            "Blood pressure is recorded as two numbers: systolic pressure "
            "over diastolic pressure, measured in millimeters of mercury. "
            "Normal blood pressure is generally below 120/80 mmHg, while "
            "readings at or above 180/120 mmHg, especially with symptoms "
            "like chest pain, shortness of breath, or vision changes, "
            "represent a hypertensive emergency requiring immediate care."
        ),
    },
    {
        "topic": "Palpitations",
        "citation": "SEED-PALPITATIONS-001",
        "text": (
            "Palpitations are sensations of a racing, pounding, or irregular "
            "heartbeat. They can be triggered by caffeine, stress, "
            "dehydration, or certain medications, and are often harmless. "
            "Palpitations accompanied by chest pain, shortness of breath, "
            "fainting, or a family history of sudden cardiac death should be "
            "evaluated promptly, as they may indicate a heart rhythm "
            "abnormality."
        ),
    },
    {
        "topic": "Wound Care and Infection Signs",
        "citation": "SEED-WOUNDCARE-001",
        "text": (
            "Most minor wounds heal well with basic cleaning and dressing "
            "changes. Signs that a wound may be infected include increasing "
            "redness, warmth, swelling, pus, worsening pain, or fever. "
            "Wounds that show spreading redness, red streaking, or that "
            "occur in someone with diabetes or a weakened immune system "
            "should be evaluated promptly to prevent serious complications."
        ),
    },
]
