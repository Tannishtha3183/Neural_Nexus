"""
BioSpan AI - Clinical Ontology & Medical Dictionary Engine (ICD-10 Integration)
Provides comprehensive clinical definitions, pathophysiology, plain English meanings,
and ICD-10 standardized diagnostic coding across all major human organ systems.
Includes live medical encyclopedia lookup for arbitrary clinical conditions.
"""

import os
import re
import difflib
import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any, Optional

class ICD10Linker:
    """
    Authoritative Medical Dictionary & Clinical Ontology Linker.
    Maps clinical terms and sequence-labeling spans to standardized ICD-10 codes,
    clinical pathophysiology, plain English translations, and DRG billing classes.
    """

    # Comprehensive Clinical Database of ICD-10 Disease Codes & Categories
    ONTOLOGY_DATABASE = [
        # =========================================================================
        # 1. ONCOLOGY & NEOPLASMS (C00 - D49)
        # =========================================================================
        {
            "code": "C34.90",
            "name": "Non-small cell lung carcinoma (NSCLC)",
            "synonyms": ["non-small cell lung cancer", "nsclc", "lung carcinoma", "non small cell lung cancer", "lung cancer", "bronchogenic carcinoma", "lung adenocarcinoma", "squamous cell lung cancer"],
            "category": "Malignant Neoplasms of Respiratory Organs",
            "clinical_notes": "Primary malignancy of pulmonary epithelial cells, accounting for ~85% of all lung cancers. Major histological subtypes include adenocarcinoma, squamous cell carcinoma, and large cell carcinoma. Requires molecular profiling for EGFR mutations, ALK rearrangements, and PD-L1 expression to direct targeted tyrosine kinase inhibitors or immunotherapy.",
            "plain_english": "The most common form of lung cancer that develops in the outer or lining tissues of the lung. Typically presents with persistent cough, breathlessness, or chest pain.",
            "billing_class": "Oncology Inpatient / Outpatient Major Diagnostic Category"
        },
        {
            "code": "C34.91",
            "name": "Small cell lung carcinoma (SCLC)",
            "synonyms": ["small cell lung cancer", "sclc", "oat cell carcinoma"],
            "category": "Malignant Neoplasms of Respiratory Organs",
            "clinical_notes": "Highly aggressive neuroendocrine tumor strongly linked to cigarette smoking. Characterized by rapid doubling time, high growth fraction, and early distant hematogenous dissemination. Staged as Limited or Extensive disease.",
            "plain_english": "A fast-growing type of lung cancer strongly associated with smoking that tends to spread early to other parts of the body.",
            "billing_class": "Oncology Major Diagnostic Category"
        },
        {
            "code": "C50.919",
            "name": "Invasive ductal carcinoma of breast",
            "synonyms": ["breast cancer", "breast carcinoma", "invasive ductal carcinoma", "ductal carcinoma", "infiltrating duct carcinoma"],
            "category": "Malignant Neoplasms of Breast",
            "clinical_notes": "Most frequent invasive breast neoplasm, originating in the milk ducts and invading surrounding stroma. Receptor status (ER, PR, and HER2/neu) and genomic profiling guide endocrine therapy vs targeted monoclonal antibody therapy.",
            "plain_english": "A common breast cancer that begins in the milk ducts and spreads into surrounding breast tissue. Detected by mammography or breast lump.",
            "billing_class": "Oncology / Surgical Coding"
        },
        {
            "code": "C71.9",
            "name": "Glioblastoma multiforme (GBM)",
            "synonyms": ["glioblastoma", "gbm", "high-grade astrocytoma", "glioblastoma multiforme", "brain tumor", "malignant glioma"],
            "category": "Malignant Neoplasms of Central Nervous System",
            "clinical_notes": "WHO Grade IV diffuse astrocytic tumor characterized by cellular polymorphism, brisk mitotic activity, vascular endothelial proliferation, and pseudopalisading necrosis. Standard of care includes surgical resection followed by concurrent temozolomide and radiotherapy.",
            "plain_english": "An aggressive, fast-growing tumor originating in the brain's supportive tissue. Symptoms include severe headaches, seizures, and focal neurological changes.",
            "billing_class": "Neurosurgical / Neuro-Oncology"
        },
        {
            "code": "C43.9",
            "name": "Malignant melanoma of skin",
            "synonyms": ["melanoma", "malignant melanoma", "cutaneous melanoma"],
            "category": "Malignant Neoplasms of Skin",
            "clinical_notes": "Malignancy of epidermal melanocytes with strong association with ultraviolet radiation. Breslow tumor depth dictates surgical excision margins and sentinel lymph node biopsy. BRAF V600E mutations common therapeutic target.",
            "plain_english": "A serious form of skin cancer that begins in the pigment-producing cells. Identified by changing, asymmetrical, or dark skin moles.",
            "billing_class": "Dermatology / Surgical Oncology"
        },
        {
            "code": "C18.9",
            "name": "Colorectal adenocarcinoma (Colon cancer)",
            "synonyms": ["colon cancer", "colorectal cancer", "colorectal carcinoma", "colon adenocarcinoma", "rectal cancer", "hereditary nonpolyposis colorectal cancer", "hnpcc", "lynch syndrome"],
            "category": "Malignant Neoplasms of Digestive Organs",
            "clinical_notes": "Malignant epithelial tumor of the colon or rectum arising via the adenoma-carcinoma sequence. Microsatellite instability (MSI/MMR) testing guides immune checkpoint inhibitor therapy.",
            "plain_english": "Cancer that starts in the large intestine or rectum, often beginning as benign polyps. Symptoms include changes in bowel habits or occult blood.",
            "billing_class": "Gastroenterology / Surgical Oncology"
        },
        {
            "code": "C91.50",
            "name": "Adult T-cell leukaemia / lymphoma",
            "synonyms": ["sporadic t - cell leukaemia", "t - cell leukaemia", "t-cell leukemia", "t cell leukaemia", "t-cell lymphoma", "sporadic t-cell leukaemia"],
            "category": "Malignant Neoplasms of Lymphoid and Hematopoietic Tissue",
            "clinical_notes": "Aggressive malignant lymphoproliferative neoplasm of mature peripheral T lymphocytes. Frequently linked with HTLV-1 retroviral infection. Manifests with circulating leukemic flower cells, hypercalcemia, and lytic bone lesions.",
            "plain_english": "A rare, fast-progressing cancer of white blood cells known as T-cells that affects blood, lymph nodes, and skin.",
            "billing_class": "Hematologic Oncology DRG"
        },
        {
            "code": "C92.00",
            "name": "Acute myeloid leukaemia (AML)",
            "synonyms": ["acute myeloid leukemia", "aml", "acute myelogenous leukemia", "acute nonlymphocytic leukemia"],
            "category": "Malignant Neoplasms of Lymphoid and Hematopoietic Tissue",
            "clinical_notes": "Clonal expansion of myeloid blasts in bone marrow (>20%) causing bone marrow failure, profound anemia, neutropenia, and thrombocytopenia.",
            "plain_english": "A rapidly developing cancer of the blood and bone marrow with excess immature white blood cells.",
            "billing_class": "Inpatient Hematology Oncology"
        },
        {
            "code": "C90.00",
            "name": "Multiple myeloma",
            "synonyms": ["multiple myeloma", "myeloma", "plasma cell myeloma", "kahler disease"],
            "category": "Malignant Neoplasms of Lymphoid and Hematopoietic Tissue",
            "clinical_notes": "Malignant clonal proliferation of plasma cells producing monoclonal paraprotein (M-spike). Defined by CRAB criteria: Hypercalcemia, Renal insufficiency, Anemia, and Bone lytic lesions.",
            "plain_english": "A cancer of plasma cells in the bone marrow that weakens bones and interferes with healthy blood cell production.",
            "billing_class": "Hematology Oncology DRG"
        },
        {
            "code": "C61",
            "name": "Malignant neoplasm of prostate",
            "synonyms": ["prostate cancer", "prostate carcinoma", "prostatic adenocarcinoma"],
            "category": "Malignant Neoplasms of Male Genital Organs",
            "clinical_notes": "Adenocarcinoma arising in the peripheral zone of the prostate gland. Staged using serum PSA, Gleason architectural grade, and clinical TNM criteria.",
            "plain_english": "Cancer of the prostate gland in men, often slow-growing and detected early by PSA blood screening or urinary hesitation.",
            "billing_class": "Urology / Surgical Oncology"
        },
        {
            "code": "C25.9",
            "name": "Pancreatic adenocarcinoma",
            "synonyms": ["pancreatic cancer", "pancreatic carcinoma", "adenocarcinoma of pancreas"],
            "category": "Malignant Neoplasms of Digestive Organs",
            "clinical_notes": "Ductal adenocarcinoma of the pancreas, frequently presenting at an advanced unresectable stage with painless obstructive jaundice and weight loss.",
            "plain_english": "An aggressive cancer in the pancreas organ, often causing digestive distress, jaundice, and rapid weight loss.",
            "billing_class": "Gastroenterology / Surgical Oncology"
        },
        {
            "code": "C79.31",
            "name": "Secondary malignant neoplasm of brain (Brain metastases)",
            "synonyms": ["brain metastases", "cerebral metastases", "brain secondary", "metastatic brain disease", "brain metastasis"],
            "category": "Secondary Malignant Neoplasms",
            "clinical_notes": "Hematogenous metastatic seeding of intracranial parenchyma from a primary extracranial malignancy, most commonly lung, breast, melanoma, or renal cell carcinoma.",
            "plain_english": "Cancer that started in another organ (like the lungs or breast) and has spread to the brain tissue.",
            "billing_class": "Neuro-Oncology Major DRG"
        },
        {
            "code": "G11.3",
            "name": "Ataxia - telangiectasia (Louis-Bar syndrome)",
            "synonyms": ["ataxia - telangiectasia", "ataxia-telangiectasia", "ataxia telangiectasia", "a - t", "at", "louis-bar syndrome"],
            "category": "Systemic Atrophies Primarily Affecting CNS (NCBI Benchmark)",
            "clinical_notes": "Autosomal recessive neurodegenerative syndrome caused by mutations in the ATM serine/threonine kinase gene. Features progressive cerebellar ataxia, oculocutaneous telangiectasias, and high cancer predisposition.",
            "plain_english": "A rare genetic disorder causing progressive loss of muscle coordination, dilated blood vessels, and immune deficiency.",
            "billing_class": "Genetic / Pediatric Neurology DRG"
        },

        # =========================================================================
        # 2. CARDIOLOGY & CIRCULATORY SYSTEM (I00 - I99)
        # =========================================================================
        {
            "code": "I21.9",
            "name": "Acute myocardial infarction, unspecified",
            "synonyms": ["myocardial infarction", "acute myocardial infarction", "heart attack", "stemi", "nstemi", "coronary thrombosis", "acute coronary syndrome"],
            "category": "Ischemic Heart Diseases",
            "clinical_notes": "Ischemic myocardial necrosis caused by acute rupture or erosion of vulnerable atherosclerotic plaque with subsequent occlusive coronary thrombosis. Diagnosed by dynamic cardiac troponin elevation with typical ischemic ECG changes.",
            "plain_english": "Commonly known as a heart attack. Occurs when a sudden clot blocks blood flow to the heart muscle, starving tissue of oxygen.",
            "billing_class": "Emergency / Critical Care DRG"
        },
        {
            "code": "I25.10",
            "name": "Coronary artery disease (Atherosclerotic heart disease)",
            "synonyms": ["coronary artery disease", "cad", "coronary heart disease", "ischemic heart disease", "atherosclerosis"],
            "category": "Chronic Ischemic Heart Diseases",
            "clinical_notes": "Progressive fibrofatty atheromatous plaque accumulation within epicardial coronary arteries, narrowing the vascular lumen and impeding myocardial perfusion.",
            "plain_english": "Narrowing of the blood vessels supplying oxygen to the heart muscle due to plaque buildup.",
            "billing_class": "Cardiology Major Diagnostic Category"
        },
        {
            "code": "I20.9",
            "name": "Angina pectoris, unspecified",
            "synonyms": ["angina", "angina pectoris", "chest pain of cardiac origin", "ischemic chest pain", "stable angina", "unstable angina"],
            "category": "Ischemic Heart Diseases",
            "clinical_notes": "Transient myocardial ischemia resulting from an imbalance between myocardial oxygen supply and demand. Characterized by retrosternal pressure radiating to the left arm or jaw, exacerbated by exertion and relieved by rest or nitroglycerin.",
            "plain_english": "Chest tightness or squeezing pain caused by reduced blood flow to the heart during activity or stress.",
            "billing_class": "Cardiovascular Evaluation"
        },
        {
            "code": "I42.0",
            "name": "Dilated cardiomyopathy",
            "synonyms": ["dilated cardiomyopathy", "dcm", "congestive cardiomyopathy"],
            "category": "Diseases of the Circulatory System (Cardiomyopathy)",
            "clinical_notes": "Ventricular chamber enlargement with impaired systolic contractile function (reduced ejection fraction). Etiologies include genetic titin mutations, chronic alcohol toxicity, viral myocarditis, or doxorubicin cardiotoxicity.",
            "plain_english": "An enlarged and weakened heart muscle that struggles to pump blood effectively throughout the body.",
            "billing_class": "Cardiology Inpatient DRG"
        },
        {
            "code": "I42.1",
            "name": "Hypertrophic obstructive cardiomyopathy",
            "synonyms": ["hypertrophic cardiomyopathy", "hocm", "hcm", "asymmetric septal hypertrophy"],
            "category": "Diseases of the Circulatory System (Cardiomyopathy)",
            "clinical_notes": "Autosomal dominant sarcomeric gene mutation causing asymmetric left ventricular hypertrophy, myofibrillar disarray, and dynamic left ventricular outflow tract (LVOT) obstruction.",
            "plain_english": "A genetic condition where the heart muscle walls become abnormally thick, making it harder for the heart to pump blood.",
            "billing_class": "Cardiology Specialist Billing"
        },
        {
            "code": "I50.9",
            "name": "Heart failure, unspecified (Congestive heart failure)",
            "synonyms": ["heart failure", "chf", "congestive heart failure", "cardiac failure", "hfref", "hfpef"],
            "category": "Heart Failure & Complications",
            "clinical_notes": "Complex clinical syndrome resulting from structural or functional cardiac impairment that prevents the ventricles from filling or ejecting blood at normal physiologic pressures.",
            "plain_english": "A chronic condition where the heart cannot pump enough blood to meet the body's daily needs, causing fluid buildup and fatigue.",
            "billing_class": "High-Volume Inpatient DRG"
        },
        {
            "code": "I48.91",
            "name": "Atrial fibrillation, unspecified",
            "synonyms": ["atrial fibrillation", "afib", "a-fib", "auricular fibrillation"],
            "category": "Cardiac Arrhythmias",
            "clinical_notes": "Supraventricular tachyarrhythmia with uncoordinated atrial activation and variable ventricular response. Carries a fivefold increased risk of thromboembolic stroke (evaluated by CHA2DS2-VASc score).",
            "plain_english": "An irregular, often very rapid heartbeat originating in the top chambers of the heart that raises the risk of blood clots and stroke.",
            "billing_class": "Cardiology Outpatient / Inpatient"
        },
        {
            "code": "I10",
            "name": "Essential (primary) hypertension",
            "synonyms": ["hypertension", "high blood pressure", "htn", "essential hypertension", "systemic hypertension"],
            "category": "Hypertensive Diseases",
            "clinical_notes": "Sustained elevation of systemic arterial blood pressure (systolic >= 130 mmHg or diastolic >= 80 mmHg) without identifiable secondary endocrine or renal etiology.",
            "plain_english": "High blood pressure in the arteries that forces the heart to work harder. Over time, it increases risks of stroke and kidney damage.",
            "billing_class": "Chronic Disease Management"
        },
        {
            "code": "I27.20",
            "name": "Pulmonary hypertension, unspecified",
            "synonyms": ["pulmonary hypertension", "pulmonary arterial hypertension", "pah", "secondary pulmonary hypertension", "mild secondary pulmonary hypertension"],
            "category": "Other Pulmonary Heart Diseases",
            "clinical_notes": "Elevated mean pulmonary arterial pressure (>20 mmHg at rest) determined via right heart catheterization. Classified into 5 WHO clinical groups.",
            "plain_english": "High blood pressure inside the arteries that travel from the heart to the lungs, making breathing feel labored during normal movement.",
            "billing_class": "Cardiopulmonary Specialty DRG"
        },
        {
            "code": "I26.99",
            "name": "Pulmonary embolism without acute cor pulmonale",
            "synonyms": ["pulmonary embolism", "pe", "pulmonary thromboembolism", "blood clot in lung"],
            "category": "Pulmonary Heart Disease & Vascular Diseases",
            "clinical_notes": "Acute occlusion of pulmonary arterial bed by a dislodged thrombus, typically originating from deep veins of the lower extremities (DVT). Causes ventilation-perfusion mismatch and acute right ventricular strain.",
            "plain_english": "A medical emergency where a blood clot (usually from the legs) travels to the lungs, suddenly blocking blood flow.",
            "billing_class": "Critical Care / Emergency DRG"
        },
        {
            "code": "I80.209",
            "name": "Deep vein thrombosis (DVT)",
            "synonyms": ["deep vein thrombosis", "dvt", "venous thrombosis", "deep venous thrombosis"],
            "category": "Diseases of Veins, Lymphatic Vessels",
            "clinical_notes": "Formation of a fibrin clot within the deep veins of the leg or pelvis. Manifests with unilateral lower extremity edema, erythema, and calf tenderness.",
            "plain_english": "A blood clot that forms in a deep vein, most commonly in the leg, causing swelling, warmth, and aching pain.",
            "billing_class": "Vascular Surgery / Medicine"
        },
        {
            "code": "I35.0",
            "name": "Nonrheumatic aortic valve stenosis",
            "synonyms": ["aortic stenosis", "as", "aortic valve stenosis", "calcific aortic stenosis"],
            "category": "Nonrheumatic Valve Disorders",
            "clinical_notes": "Pathologic narrowing of the aortic valve orifice, predominantly due to age-related dystrophic calcification or a congenital bicuspid valve.",
            "plain_english": "Narrowing of the heart's exit valve, restricting blood flow from the heart to the rest of the body.",
            "billing_class": "Cardiology / Cardiothoracic Surgery"
        },
        {
            "code": "R00.0",
            "name": "Sinus tachycardia, unspecified",
            "synonyms": ["sinus tachycardia", "tachycardia", "fast heart rate", "rapid pulse", "palpitations"],
            "category": "Symptoms and Signs Involving Circulatory System",
            "clinical_notes": "Sinoatrial node pacemaker firing at a rate exceeding 100 beats per minute. Physiologic in exercise or stress; pathologic in fever, hypovolemia, anemia, or thyrotoxicosis.",
            "plain_english": "A fast but regular heart rhythm above 100 beats per minute, often felt as racing pulse or fluttering in the chest.",
            "billing_class": "Cardiology Outpatient Evaluation"
        },

        # =========================================================================
        # 3. PULMONOLOGY & RESPIRATORY SYSTEM (J00 - J99)
        # =========================================================================
        {
            "code": "J45.909",
            "name": "Unspecified asthma, uncomplicated",
            "synonyms": ["asthma", "bronchial asthma", "allergic asthma", "severe asthma", "acute asthma", "exercise-induced asthma"],
            "category": "Chronic Lower Respiratory Diseases",
            "clinical_notes": "Heterogeneous chronic inflammatory disorder of the conducting airways characterized by bronchial hyperresponsiveness, reversible airflow obstruction, and airway remodeling.",
            "plain_english": "A chronic airway condition that causes coughing, wheezing, chest tightness, and shortness of breath when airways narrow.",
            "billing_class": "Pulmonology Outpatient / Emergency"
        },
        {
            "code": "J44.9",
            "name": "Chronic obstructive pulmonary disease, unspecified",
            "synonyms": ["copd", "chronic obstructive pulmonary disease", "chronic obstructive lung disease", "emphysema", "chronic bronchitis"],
            "category": "Chronic Lower Respiratory Diseases",
            "clinical_notes": "Persistent, progressive airflow limitation associated with enhanced chronic inflammatory response in airways and lung parenchyma to noxious particles or gases (primarily tobacco smoke).",
            "plain_english": "A group of progressive lung diseases (including emphysema and chronic bronchitis) that make it hard to breathe.",
            "billing_class": "Pulmonology Major DRG"
        },
        {
            "code": "J15.9",
            "name": "Bacterial pneumonia, unspecified",
            "synonyms": ["pneumonia", "bacterial pneumonia", "community acquired pneumonia", "cap", "aspiration pneumonia"],
            "category": "Respiratory Infections",
            "clinical_notes": "Acute infection of pulmonary alveolar parenchyma with inflammatory exudate, resulting in alveolar consolidation. Common pathogens include Streptococcus pneumoniae and atypical organisms.",
            "plain_english": "An infection in one or both lungs where air sacs fill with fluid or pus, causing cough with phlegm, fever, and breathing difficulty.",
            "billing_class": "Infectious Disease / Inpatient"
        },
        {
            "code": "J84.112",
            "name": "Idiopathic pulmonary fibrosis",
            "synonyms": ["idiopathic pulmonary fibrosis", "ipf", "pulmonary fibrosis", "fibrosing alveolitis"],
            "category": "Other Interstitial Pulmonary Diseases",
            "clinical_notes": "Chronic, progressive fibrosing interstitial pneumonia of unknown etiology, characterized by the radiologic and histologic pattern of usual interstitial pneumonia (UIP).",
            "plain_english": "A progressive disease where lung tissue becomes scarred, thick, and stiff over time, making it increasingly difficult to get oxygen.",
            "billing_class": "Specialty Pulmonology"
        },
        {
            "code": "J20.9",
            "name": "Acute bronchitis, unspecified",
            "synonyms": ["acute bronchitis", "bronchitis", "tracheobronchitis"],
            "category": "Acute Respiratory Infections",
            "clinical_notes": "Transient inflammation of the trachea and large bronchi, predominantly of viral etiology (rhinovirus, influenza, RSV). Manifests with cough lasting 1 to 3 weeks.",
            "plain_english": "A short-term inflammation of the main airways to the lungs, typically following a cold and causing a chesty cough.",
            "billing_class": "Primary Care / Urgent Care"
        },
        {
            "code": "J96.00",
            "name": "Acute respiratory failure, unspecified",
            "synonyms": ["acute respiratory failure", "respiratory failure", "respiratory distress", "hypoxemic respiratory failure", "hypercapnic respiratory failure"],
            "category": "Respiratory Failure",
            "clinical_notes": "Inability of the respiratory system to maintain adequate arterial oxygenation (PaO2 < 60 mmHg) or carbon dioxide elimination (PaCO2 > 50 mmHg with acidosis).",
            "plain_english": "A critical medical condition where the lungs cannot pass enough oxygen into the bloodstream or remove carbon dioxide.",
            "billing_class": "Critical Care Inpatient DRG"
        },

        # =========================================================================
        # 4. NEUROLOGY & BRAIN DISORDERS (G00 - G99, I60 - I69)
        # =========================================================================
        {
            "code": "I63.9",
            "name": "Cerebral infarction, unspecified (Ischemic stroke)",
            "synonyms": ["stroke", "ischemic stroke", "cerebral infarction", "cva", "cerebrovascular accident", "acute ischemic stroke"],
            "category": "Cerebrovascular Diseases",
            "clinical_notes": "Acute neurological deficit caused by focal cerebral ischemia from thromboembolic or atherothrombotic arterial occlusion. Time-sensitive intervention includes intravenous thrombolysis (tPA) or mechanical thrombectomy.",
            "plain_english": "Commonly known as a stroke. Occurs when blood supply to part of the brain is cut off by a clot, damaging brain cells.",
            "billing_class": "Acute Neurology DRG"
        },
        {
            "code": "G93.6",
            "name": "Cerebral edema",
            "synonyms": ["cerebral edema", "brain edema", "brain swelling", "vasogenic edema", "cytotoxic edema"],
            "category": "Other Disorders of the Brain",
            "clinical_notes": "Excessive accumulation of fluid in the intracellular (cytotoxic) or extracellular (vasogenic) compartments of the brain, leading to increased intracranial pressure (ICP) and risk of herniation.",
            "plain_english": "Swelling in the brain caused by excess fluid accumulation. Can raise pressure inside the skull, leading to headaches and altered consciousness.",
            "billing_class": "Neurosurgical / Critical Care DRG"
        },
        {
            "code": "G40.909",
            "name": "Epilepsy, unspecified, not intractable",
            "synonyms": ["epilepsy", "seizure disorder", "convulsions", "seizures", "tonic clonic seizure", "focal seizures", "epileptic seizure"],
            "category": "Episodic and Paroxysmal Disorders",
            "clinical_notes": "Chronic neurological disorder characterized by recurrent, unprovoked seizures resulting from abnormal, excessive hypersynchronous neuronal electrical activity in the cerebral cortex.",
            "plain_english": "A brain condition that causes repeated, sudden seizures or convulsions due to bursts of electrical activity in the brain.",
            "billing_class": "Neurology Outpatient / Emergency"
        },
        {
            "code": "G35",
            "name": "Multiple sclerosis",
            "synonyms": ["multiple sclerosis", "ms", "relapsing remitting multiple sclerosis", "rrms", "secondary progressive ms"],
            "category": "Demyelinating Diseases of Central Nervous System",
            "clinical_notes": "Chronic autoimmune inflammatory demyelinating disease of the central nervous system. Features optic neuritis, sensory deficits, weakness, and MRI dissemination in space and time.",
            "plain_english": "An autoimmune condition where the body's immune system attacks the protective covering of nerves in the brain and spinal cord.",
            "billing_class": "Neurology Major Diagnostic Category"
        },
        {
            "code": "G30.9",
            "name": "Alzheimer's disease, unspecified",
            "synonyms": ["alzheimer's disease", "alzheimers", "ad", "dementia of alzheimer type"],
            "category": "Neurodegenerative Diseases",
            "clinical_notes": "Neurodegenerative disorder characterized by extracellular amyloid-beta plaque deposition and intracellular hyperphosphorylated tau neurofibrillary tangles, leading to progressive cognitive decline.",
            "plain_english": "A progressive brain disease that slowly destroys memory and thinking skills, being the most common cause of dementia.",
            "billing_class": "Geriatric Neurology DRG"
        },
        {
            "code": "G20",
            "name": "Parkinson's disease",
            "synonyms": ["parkinson's disease", "parkinsons", "pd", "paralysis agitans"],
            "category": "Extrapyramidal and Movement Disorders",
            "clinical_notes": "Progressive neurodegenerative condition caused by loss of dopaminergic neurons in the substantia nigra pars compacta. Cardinal signs: resting tremor, rigidity, bradykinesia, and postural instability.",
            "plain_english": "A progressive movement disorder caused by the loss of dopamine-producing cells in the brain, leading to tremors, stiffness, and slowed motion.",
            "billing_class": "Neurology Chronic Care"
        },
        {
            "code": "G62.9",
            "name": "Peripheral neuropathy, unspecified",
            "synonyms": ["peripheral neuropathy", "polyneuropathy", "neuropathy", "diabetic neuropathy", "diabetic peripheral neuropathy"],
            "category": "Diseases of the Nervous System",
            "clinical_notes": "Damage to the peripheral nervous system, most commonly from diabetic microvascular ischemia, vitamin B12 deficiency, or toxic chemotherapy. Manifests with distal symmetric 'stocking-glove' sensory loss and paresthesias.",
            "plain_english": "Nerve damage outside the brain and spinal cord that causes numbness, tingling, weakness, and burning pain, usually in the feet and hands.",
            "billing_class": "Neurology Outpatient"
        },
        {
            "code": "G12.21",
            "name": "Amyotrophic lateral sclerosis (ALS)",
            "synonyms": ["amyotrophic lateral sclerosis", "als", "lou gehrig disease", "motor neuron disease"],
            "category": "Systemic Atrophies Affecting CNS",
            "clinical_notes": "Fatal neurodegenerative disorder involving both upper motor neurons (hyperreflexia, spasticity) and lower motor neurons (muscle atrophy, fasciculations), culminating in respiratory failure.",
            "plain_english": "A progressive nervous system disease that attacks nerve cells in the brain and spinal cord, causing loss of muscle control.",
            "billing_class": "Neurology Inpatient / Outpatient"
        },
        {
            "code": "G43.909",
            "name": "Migraine, unspecified, not intractable",
            "synonyms": ["migraine", "migraine headache", "hemicrania", "migraine with aura"],
            "category": "Headache Syndromes",
            "clinical_notes": "Complex neurovascular disorder characterized by recurrent, pulsating, unilateral headaches lasting 4 to 72 hours, aggravated by routine physical activity and accompanied by photophobia, phonophobia, or nausea.",
            "plain_english": "A severe, throbbing headache typically on one side of the head, often accompanied by sensitivity to light and sound, or nausea.",
            "billing_class": "Neurology Outpatient"
        },
        {
            "code": "R51.9",
            "name": "Headache, unspecified",
            "synonyms": ["headache", "cephalea", "head pain", "tension headache", "cranial pain"],
            "category": "Symptoms Involving Nervous System",
            "clinical_notes": "Pain located in the cranial vault or upper cervical region. Must be differentiated into primary headaches (tension, migraine) vs secondary red-flag causes (subarachnoid hemorrhage, meningitis, mass lesion).",
            "plain_english": "Pain anywhere in the region of the head or neck, ranging from mild tension aches to severe underlying neurological pressure.",
            "billing_class": "Neurology Diagnostic Code"
        },

        # =========================================================================
        # 5. ENDOCRINOLOGY & METABOLIC DISORDERS (E00 - E89)
        # =========================================================================
        {
            "code": "E11.9",
            "name": "Type 2 diabetes mellitus without complications",
            "synonyms": ["type 2 diabetes", "t2d", "type 2 diabetes mellitus", "t2dm", "adult-onset diabetes", "non-insulin-dependent diabetes"],
            "category": "Endocrine, Nutritional and Metabolic Diseases",
            "clinical_notes": "Metabolic disorder characterized by peripheral insulin resistance combined with progressive pancreatic beta-cell secretory defect, leading to chronic systemic hyperglycemia.",
            "plain_english": "A chronic condition that affects how the body processes blood sugar (glucose), resulting in high sugar levels if untreated.",
            "billing_class": "Endocrinology Chronic Care"
        },
        {
            "code": "E11.40",
            "name": "Type 2 diabetes mellitus with diabetic neuropathy",
            "synonyms": ["diabetic neuropathy", "diabetic peripheral neuropathy", "t2dm with neuropathy", "type 2 diabetes with neuropathy"],
            "category": "Endocrine, Nutritional and Metabolic Diseases",
            "clinical_notes": "Distal symmetric polyneuropathy resulting from chronic hyperglycemia, advanced glycation end-products, and microvascular endoneurial ischemia in diabetic patients.",
            "plain_english": "Nerve damage in the legs, feet, or hands caused by long-term high blood sugar levels from diabetes.",
            "billing_class": "Endocrinology Specialist"
        },
        {
            "code": "E11.10",
            "name": "Type 2 diabetes mellitus with ketoacidosis",
            "synonyms": ["diabetic ketoacidosis", "dka", "ketoacidosis"],
            "category": "Endocrine, Nutritional and Metabolic Diseases",
            "clinical_notes": "Life-threatening acute metabolic emergency characterized by triad of hyperglycemia, metabolic anion-gap acidosis, and hyperketonemia due to profound absolute or relative insulin deficiency.",
            "plain_english": "A dangerous diabetes emergency where the body runs out of insulin and breaks down fats too quickly, building up harmful blood acids called ketones.",
            "billing_class": "Endocrinology Emergency DRG"
        },
        {
            "code": "E10.9",
            "name": "Type 1 diabetes mellitus without complications",
            "synonyms": ["type 1 diabetes", "t1d", "t1dm", "juvenile diabetes", "insulin-dependent diabetes"],
            "category": "Endocrine, Nutritional and Metabolic Diseases",
            "clinical_notes": "Autoimmune destruction of pancreatic insulin-producing beta cells mediated by anti-GAD65 and anti-islet autoantibodies, resulting in complete lifelong insulin deficiency.",
            "plain_english": "An autoimmune condition where the pancreas produces little or no insulin, requiring daily insulin therapy.",
            "billing_class": "Endocrinology Inpatient / Outpatient"
        },
        {
            "code": "E03.9",
            "name": "Hypothyroidism, unspecified",
            "synonyms": ["hypothyroidism", "underactive thyroid", "hashimoto's thyroiditis", "hashimoto disease", "myxedema"],
            "category": "Thyroid Disorders",
            "clinical_notes": "Deficiency of thyroid hormones (free T4 and T3) with elevated TSH. Most commonly autoimmune Hashimoto thyroiditis in iodine-sufficient areas. Symptoms include fatigue, weight gain, cold intolerance, and constipation.",
            "plain_english": "A condition where the thyroid gland does not produce enough thyroid hormones, slowing down the body's metabolism.",
            "billing_class": "Endocrinology Outpatient"
        },
        {
            "code": "E05.90",
            "name": "Thyrotoxicosis, unspecified (Hyperthyroidism)",
            "synonyms": ["hyperthyroidism", "thyrotoxicosis", "graves' disease", "overactive thyroid"],
            "category": "Thyroid Disorders",
            "clinical_notes": "Excessive concentrations of circulating thyroid hormones. Graves disease is the leading cause, driven by stimulating TSH-receptor autoantibodies producing goiter and exophthalmos.",
            "plain_english": "An overactive thyroid gland producing too much hormone, causing weight loss, rapid heartbeat, sweating, and anxiety.",
            "billing_class": "Endocrinology Outpatient"
        },
        {
            "code": "E78.01",
            "name": "Familial hypercholesterolemia",
            "synonyms": ["familial hypercholesterolemia", "hypercholesterolemia", "hyperlipidemia", "dyslipidemia", "high cholesterol"],
            "category": "Metabolic Disorders",
            "clinical_notes": "Monogenic disorder of low-density lipoprotein (LDL) clearance (primarily LDL receptor mutations) causing severe lifelong hypercholesterolemia and premature coronary artery disease.",
            "plain_english": "An inherited condition that causes very high levels of cholesterol in the blood, greatly increasing the risk of early heart disease.",
            "billing_class": "Preventive Cardiology / Genetics"
        },

        # =========================================================================
        # 6. GASTROENTEROLOGY & HEPATOLOGY (K00 - K95)
        # =========================================================================
        {
            "code": "K21.9",
            "name": "Gastro-esophageal reflux disease without esophagitis",
            "synonyms": ["gerd", "acid reflux", "gastroesophageal reflux", "heartburn", "reflux esophagitis"],
            "category": "Diseases of the Digestive System",
            "clinical_notes": "Incompetence of the lower esophageal sphincter allowing retrograde flow of acidic gastric content into the esophagus, causing retrosternal burning and mucosal erosion.",
            "plain_english": "A digestive condition where stomach acid frequently flows back into the tube connecting the mouth and stomach, causing heartburn.",
            "billing_class": "Gastroenterology Outpatient"
        },
        {
            "code": "K50.90",
            "name": "Crohn's disease, unspecified",
            "synonyms": ["crohn's disease", "crohns", "regional enteritis", "inflammatory bowel disease", "ibd"],
            "category": "Noninfective Enteritis and Colitis",
            "clinical_notes": "Chronic transmural granulomatous inflammatory disorder of the gastrointestinal tract capable of affecting any segment from mouth to anus, typically with 'skip lesions' and fistulization.",
            "plain_english": "A chronic inflammatory bowel disease that causes ongoing inflammation and ulcers anywhere along the digestive tract.",
            "billing_class": "Gastroenterology Inpatient / Outpatient"
        },
        {
            "code": "K51.90",
            "name": "Ulcerative colitis, unspecified",
            "synonyms": ["ulcerative colitis", "uc", "colitis", "ulcerative proctitis"],
            "category": "Noninfective Enteritis and Colitis",
            "clinical_notes": "Chronic mucosal inflammatory disease restricted to the colon and rectum, beginning distally and progressing proximally in a continuous, uninterrupted pattern with bloody diarrhea.",
            "plain_english": "An inflammatory bowel disease that causes long-lasting inflammation and ulcers in the innermost lining of the large intestine and rectum.",
            "billing_class": "Gastroenterology Specialist"
        },
        {
            "code": "K74.60",
            "name": "Unspecified cirrhosis of liver",
            "synonyms": ["cirrhosis", "liver cirrhosis", "hepatic cirrhosis", "end-stage liver disease", "hepatic fibrosis"],
            "category": "Diseases of the Liver",
            "clinical_notes": "Late-stage hepatic fibrosis with distortion of vascular architecture and regenerative nodule formation. Complications include portal hypertension, ascites, esophageal varices, and hepatic encephalopathy.",
            "plain_english": "Severe, irreversible scarring of the liver caused by long-term damage, impairing the liver's ability to filter toxins.",
            "billing_class": "Hepatology Major DRG"
        },
        {
            "code": "K85.90",
            "name": "Acute pancreatitis, unspecified",
            "synonyms": ["acute pancreatitis", "pancreatitis", "pancreatic inflammation"],
            "category": "Diseases of the Pancreas",
            "clinical_notes": "Acute autodigestive inflammatory condition of the pancreas resulting from premature intracellular activation of trypsinogen. Major causes are gallstones and alcohol abuse. Hallmark epigastric pain radiating to back.",
            "plain_english": "Sudden inflammation of the pancreas gland causing severe upper belly pain that radiates into the back, along with nausea and vomiting.",
            "billing_class": "Gastroenterology Emergency DRG"
        },

        # =========================================================================
        # 7. NEPHROLOGY & RENAL SYSTEM (N00 - N39)
        # =========================================================================
        {
            "code": "N18.9",
            "name": "Chronic kidney disease, unspecified",
            "synonyms": ["chronic kidney disease", "ckd", "chronic renal disease", "renal failure", "kidney failure", "esrd", "end stage renal disease"],
            "category": "Diseases of the Urinary System",
            "clinical_notes": "Progressive, irreversible loss of renal nephron architecture and excretory filtration capacity lasting >3 months, staged by glomerular filtration rate (eGFR) and albuminuria.",
            "plain_english": "A gradual, long-term loss of kidney function over time, preventing the kidneys from filtering waste products from the blood.",
            "billing_class": "Nephrology Chronic Care"
        },
        {
            "code": "N17.9",
            "name": "Acute kidney injury, unspecified",
            "synonyms": ["acute kidney injury", "aki", "acute renal failure", "arf", "acute tubular necrosis"],
            "category": "Acute Renal Failure",
            "clinical_notes": "Abrupt decline in renal filtration function occurring within hours to days, characterized by accumulation of nitrogenous waste products (BUN, creatinine) and oliguria (KDIGO criteria).",
            "plain_english": "A sudden episode of kidney failure or kidney damage that happens within a few days, causing waste buildup in the body.",
            "billing_class": "Inpatient Nephrology DRG"
        },
        {
            "code": "N39.0",
            "name": "Urinary tract infection, site not specified",
            "synonyms": ["urinary tract infection", "uti", "cystitis", "bladder infection", "pyelonephritis"],
            "category": "Other Diseases of Urinary System",
            "clinical_notes": "Bacterial colonization of the urinary epithelium (most commonly uropathogenic Escherichia coli). Manifests with dysuria, urinary frequency, urgency, and suprapubic pain.",
            "plain_english": "An infection in any part of the urinary system (bladder or kidneys), causing painful or burning urination and urgent trips to the bathroom.",
            "billing_class": "Primary Care / Urgent Care"
        },

        # =========================================================================
        # 8. HEMATOLOGY & AUTOIMMUNE / RHEUMATOLOGY (D50 - D89, M00 - M99)
        # =========================================================================
        {
            "code": "D59.10",
            "name": "Autoimmune hemolytic anemia, unspecified",
            "synonyms": ["autoimmune hemolytic anemia", "aiha", "hemolytic anemia", "warm aiha", "cold agglutinin disease"],
            "category": "Aplastic and Other Anemias",
            "clinical_notes": "Accelerated erythrocyte destruction mediated by autoantibodies against red cell surface antigens. Diagnosed by positive Direct Antiglobulin Test (Coombs test) and elevated LDH/reticulocytes.",
            "plain_english": "A rare immune condition where the body's antibodies attack and destroy its own red blood cells faster than they can be made.",
            "billing_class": "Hematology Specialist"
        },
        {
            "code": "D69.3",
            "name": "Immune thrombocytopenic purpura",
            "synonyms": ["immune thrombocytopenia", "itp", "idiopathic thrombocytopenic purpura", "thrombocytopenia", "low platelets"],
            "category": "Hemorrhagic Conditions",
            "clinical_notes": "Isolated low platelet count (<100,000/uL) resulting from autoantibody-mediated platelet clearance by splenic macrophages and impaired thrombopoiesis.",
            "plain_english": "A blood disorder characterized by a low platelet count, leading to easy bruising, petechiae (pinpoint red spots), and bleeding.",
            "billing_class": "Hematology Outpatient"
        },
        {
            "code": "M32.9",
            "name": "Systemic lupus erythematosus, unspecified",
            "synonyms": ["systemic lupus erythematosus", "sle", "lupus", "lupus nephritis"],
            "category": "Systemic Connective Tissue Disorders",
            "clinical_notes": "Multisystem chronic autoimmune disease featuring polyclonal B-cell hyperactivity and antinuclear antibodies (ANA, anti-dsDNA), causing immune complex deposition across organs.",
            "plain_english": "A chronic autoimmune disease where the immune system mistakenly attacks healthy tissues in the skin, joints, kidneys, and heart.",
            "billing_class": "Rheumatology Specialist"
        },
        {
            "code": "M06.9",
            "name": "Rheumatoid arthritis, unspecified",
            "synonyms": ["rheumatoid arthritis", "ra", "inflammatory arthritis", "rheumatoid disease"],
            "category": "Inflammatory Polyarthropathies",
            "clinical_notes": "Chronic symmetric inflammatory polyarthritis characterized by synovial inflammation, pannus formation, and bone erosions, with positive rheumatoid factor (RF) or anti-CCP.",
            "plain_english": "An autoimmune disorder causing painful swelling and stiffness in the joints of both hands and feet, which can lead to joint deformity.",
            "billing_class": "Rheumatology Outpatient"
        },

        # =========================================================================
        # 9. INFECTIOUS DISEASES & SEPSIS (A00 - B99)
        # =========================================================================
        {
            "code": "A41.9",
            "name": "Sepsis, unspecified organism",
            "synonyms": ["sepsis", "septic shock", "septicemia", "blood poisoning", "severe sepsis"],
            "category": "Infectious and Parasitic Diseases",
            "clinical_notes": "Life-threatening organ dysfunction caused by a dysregulated host systemic response to infection (SOFA score >= 2). Septic shock involves refractory hypotension requiring vasopressors.",
            "plain_english": "A life-threatening medical emergency where the body's immune response to an infection triggers widespread inflammation and organ failure.",
            "billing_class": "Critical Care DRG"
        },
        {
            "code": "A15.0",
            "name": "Tuberculosis of lung",
            "synonyms": ["tuberculosis", "tb", "pulmonary tuberculosis", "phthisis"],
            "category": "Mycobacterial Infections",
            "clinical_notes": "Chronic granulomatous airborne infection caused by Mycobacterium tuberculosis, primarily affecting lungs with caseating granulomas. Manifests with night sweats, hemoptysis, and weight loss.",
            "plain_english": "A contagious bacterial infection of the lungs causing chronic cough, blood-tinged sputum, fever, night sweats, and weight loss.",
            "billing_class": "Infectious Disease Specialist"
        },

        # =========================================================================
        # 10. CLINICAL SIGNS & PHENOTYPIC SYMPTOMS (R00 - R69)
        # =========================================================================
        {
            "code": "R05.9",
            "name": "Cough, unspecified",
            "synonyms": ["cough", "chronic cough", "persistent cough", "dry cough", "productive cough", "hacking cough"],
            "category": "Symptoms Involving the Respiratory System",
            "clinical_notes": "Protective reflex to clear the airways. Chronic duration (>8 weeks) warrants evaluation for bronchial, cardiac, or neoplastic etiology.",
            "plain_english": "A natural reflex that clears the airways of irritants and mucus. Chronic coughing can signal lung or heart conditions.",
            "billing_class": "Primary Care / Outpatient Diagnostic"
        },
        {
            "code": "R06.02",
            "name": "Shortness of breath (Dyspnea on exertion)",
            "synonyms": ["shortness of breath", "dyspnea", "breathlessness", "dyspnea on exertion", "labored breathing"],
            "category": "Symptoms Involving the Respiratory System",
            "clinical_notes": "Subjective experience of breathing discomfort; cardinal presentation of cardiopulmonary pathology.",
            "plain_english": "A feeling of breathlessness or difficult, labored breathing during exertion or rest.",
            "billing_class": "Evaluation & Management (E&M)"
        },
        {
            "code": "R53.83",
            "name": "Other fatigue (Chronic fatigue and malaise)",
            "synonyms": ["fatigue", "chronic fatigue", "malaise", "tiredness", "exhaustion", "lethargy"],
            "category": "General Symptoms and Signs",
            "clinical_notes": "Persistent subjective state of physical and mental exhaustion not relieved by sleep. Common constitutional sign in oncologic, hematologic, or autoimmune conditions.",
            "plain_english": "Extreme ongoing tiredness and lack of energy that does not improve with rest.",
            "billing_class": "General Clinical Evaluation"
        },
        {
            "code": "R04.2",
            "name": "Hemoptysis",
            "synonyms": ["hemoptysis", "coughing blood", "blood-tinged sputum"],
            "category": "Symptoms Involving the Respiratory System",
            "clinical_notes": "Expectoration of blood originating from lower respiratory tract. Common red-flag sign for bronchogenic carcinoma, pulmonary embolism, or tuberculosis.",
            "plain_english": "Coughing up blood or blood-stained mucus from the lungs or airways.",
            "billing_class": "Diagnostic Symptom Code"
        },
        {
            "code": "R56.9",
            "name": "Unspecified convulsions / Seizures",
            "synonyms": ["seizures", "convulsions", "epileptic seizure", "seizure disorder", "epilepsy", "focal seizures", "tonic clonic seizure"],
            "category": "Symptoms Involving Nervous System",
            "clinical_notes": "Paroxysmal clinical event caused by abnormal synchronous electrical discharges in cortical neurons.",
            "plain_english": "Sudden, uncontrolled electrical disturbances in the brain causing jerking movements or altered awareness.",
            "billing_class": "Neurology Emergency DRG"
        },
        {
            "code": "R50.9",
            "name": "Fever, unspecified",
            "synonyms": ["fever", "pyrexia", "febrile", "hyperthermia"],
            "category": "General Symptoms and Signs",
            "clinical_notes": "Elevation of body temperature above normal circadian range due to cytokine-mediated hypothalamic set-point elevation.",
            "plain_english": "A temporary increase in body temperature, often due to an infection as part of the body's defense.",
            "billing_class": "General Clinical Observation"
        },
        {
            "code": "R64",
            "name": "Cachexia",
            "synonyms": ["cachexia", "wasting syndrome", "cancer cachexia", "weight loss"],
            "category": "General Symptoms and Signs",
            "clinical_notes": "Complex metabolic syndrome associated with underlying illness and characterized by loss of muscle with or without loss of fat mass.",
            "plain_english": "Severe unintended muscle wasting and weight loss caused by a chronic illness like cancer or heart failure.",
            "billing_class": "Nutritional & Chronic Disease Modifier"
        }
    ]

    def __init__(self):
        self.external_cache = {}

    def lookup_external(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Queries authoritative medical encyclopedia REST API (Wikipedia / NIH summary)
        for any medical condition not already in the pre-compiled database.
        Caches results in memory for high-speed repeated access.
        """
        clean = query.strip()
        if not clean:
            return None
        
        cache_key = clean.lower()
        if cache_key in self.external_cache:
            return self.external_cache[cache_key]

        slug = clean.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(slug)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BioSpanAI/1.0 (Biomedical Clinical NER Research)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                title = data.get("title", clean)
                extract = data.get("extract", "")
                description = data.get("description", "Medical Condition / Clinical Syndrome")
                
                # If extract is valid, formulate entry
                if extract:
                    first_sent = extract.split(". ")[0] + "." if ". " in extract else extract
                    entry = {
                        "code": "R69.X",
                        "name": title,
                        "synonyms": [clean.lower(), title.lower()],
                        "category": f"Clinical Medicine / {description}",
                        "clinical_notes": extract,
                        "plain_english": first_sent,
                        "billing_class": "Clinical Diagnostic Evaluation",
                        "match_confidence": 0.90,
                        "match_type": "LIVE_ENCYCLOPEDIA"
                    }
                    self.external_cache[cache_key] = entry
                    return entry
        except Exception:
            pass

        return None

    def search_codes(self, query: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches the medical dictionary by query and/or category.
        If local search returns 0 matches for a query, automatically queries live medical definition.
        """
        results = list(self.ONTOLOGY_DATABASE)

        # 1. Filter by category
        if category and category.lower() != "all":
            cat_lower = category.lower()
            results = [c for c in results if cat_lower in c.get("category", "").lower()]

        # 2. Filter by search query
        if query:
            q_clean = query.strip().lower()
            q_norm = re.sub(r'[^\w\s]', '', q_clean)
            
            matched = []
            for item in results:
                name_norm = re.sub(r'[^\w\s]', '', item["name"].lower())
                code_norm = item["code"].lower()
                syns = [re.sub(r'[^\w\s]', '', s.lower()) for s in item.get("synonyms", [])]
                notes = item.get("clinical_notes", "").lower()
                plain = item.get("plain_english", "").lower()

                if (q_norm in name_norm or 
                    q_clean in code_norm or 
                    any(q_norm in s for s in syns) or 
                    q_norm in notes or 
                    q_norm in plain):
                    matched.append(item)
            
            # If local search has matches, return them
            if matched:
                return matched

            # If local search had 0 matches, perform live medical lookup!
            live_item = self.lookup_external(query)
            if live_item:
                return [live_item]

            return []

        return results

    def link_entity(self, entity_text: str) -> Dict[str, Any]:
        """
        Takes raw entity text (e.g. 'non-small cell lung carcinoma', 'asthma', 'dilated cardiomyopathy')
        and maps to the best matching ICD-10 ontology entry via multi-faceted similarity.
        Falls back to live medical encyclopedia if unmatched locally.
        """
        clean_text = entity_text.strip().lower()
        clean_text_norm = re.sub(r'[^\w\s]', '', clean_text)

        best_match = None
        highest_score = 0.0

        for entry in self.ONTOLOGY_DATABASE:
            # 1. Exact name match
            entry_name_norm = re.sub(r'[^\w\s]', '', entry["name"].lower())
            if clean_text_norm == entry_name_norm:
                return self._format_result(entry, confidence=1.0, match_type="EXACT_NAME")

            # 2. Exact synonym match
            for syn in entry["synonyms"]:
                syn_norm = re.sub(r'[^\w\s]', '', syn.lower())
                if clean_text_norm == syn_norm:
                    return self._format_result(entry, confidence=0.98, match_type="EXACT_SYNONYM")

            # 3. Substring containment match
            for syn in entry["synonyms"]:
                syn_norm = re.sub(r'[^\w\s]', '', syn.lower())
                if clean_text_norm in syn_norm or syn_norm in clean_text_norm:
                    len_ratio = min(len(clean_text_norm), len(syn_norm)) / max(len(clean_text_norm), len(syn_norm))
                    score = 0.85 + (len_ratio * 0.1)
                    if score > highest_score:
                        highest_score = score
                        best_match = (entry, score, "SUBSTRING_CONTAINMENT")

            # 4. Levenshtein ratio matching
            sim_name = difflib.SequenceMatcher(None, clean_text_norm, entry_name_norm).ratio()
            if sim_name > highest_score:
                highest_score = sim_name
                best_match = (entry, sim_name, "FUZZY_NAME")

            for syn in entry["synonyms"]:
                sim_syn = difflib.SequenceMatcher(None, clean_text_norm, syn.lower()).ratio()
                if sim_syn > highest_score:
                    highest_score = sim_syn
                    best_match = (entry, sim_syn, "FUZZY_SYNONYM")

        if best_match and highest_score >= 0.55:
            entry, score, match_type = best_match
            return self._format_result(entry, confidence=round(score, 3), match_type=match_type)

        # Attempt live encyclopedia lookup before fallback
        live_entry = self.lookup_external(entity_text)
        if live_entry:
            return live_entry

        # Fallback for unmapped clinical entities
        return {
            "query_text": entity_text,
            "code": "R69",
            "name": f"Clinical Finding ({entity_text})",
            "category": "General Symptoms, Signs and Abnormal Clinical Findings",
            "clinical_notes": "Entity flagged by BioBERT NER with disease boundary semantics. Clinical evaluation recommended.",
            "plain_english": f"A medical condition or phenotype mention ({entity_text}) identified in clinical documentation.",
            "billing_class": "Unspecified Clinical Finding",
            "match_confidence": 0.50,
            "match_type": "FALLBACK_GENERAL"
        }

    def _format_result(self, entry: Dict[str, Any], confidence: float, match_type: str) -> Dict[str, Any]:
        return {
            "code": entry["code"],
            "name": entry["name"],
            "category": entry["category"],
            "clinical_notes": entry["clinical_notes"],
            "plain_english": entry.get("plain_english", entry["clinical_notes"]),
            "billing_class": entry["billing_class"],
            "match_confidence": confidence,
            "match_type": match_type
        }

    def get_all_codes(self) -> List[Dict[str, Any]]:
        return self.ONTOLOGY_DATABASE
