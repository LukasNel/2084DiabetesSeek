import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm.auto import tqdm
from datasets import Dataset
import duckdb

class ClinicalDataProcessor:
    def __init__(self, data_path: str):
        """Initialize the processor with path to clinical data."""
        # Initialize DuckDB connection
        self.con = duckdb.connect(database=':memory:')
        
        # Create table from CSV
        self.con.execute(f"""
            CREATE TABLE clinical_data AS 
            SELECT *,
                   TRY_CAST(admittime AS TIMESTAMP) as admittime_ts,
                   TRY_CAST(dischtime AS TIMESTAMP) as dischtime_ts,
                   TRY_CAST(deathtime AS TIMESTAMP) as deathtime_ts,
                   TRY_CAST(edregtime AS TIMESTAMP) as edregtime_ts,
                   TRY_CAST(edouttime AS TIMESTAMP) as edouttime_ts
            FROM read_csv_auto('{data_path}')
        """)
        
        self.diabetes_codes = ['250', '250.0', 'E11']  # ICD-9 and ICD-10 codes for diabetes
        
    def create_patient_timelines(self) -> Dict[str, List[Dict]]:
        """Generate chronological sequence of diagnoses for each patient."""
        # Query to get all events sorted by patient and timestamp
        query = """
            SELECT 
                subject_id,
                admittime_ts as timestamp,
                icd_code,
                diagnosis_description,
                hadm_id as admission_id
            FROM clinical_data
            ORDER BY subject_id, admittime_ts
        """
        
        results = self.con.execute(query).fetchdf()
        timelines = {}
        for _, row in tqdm(results.iterrows(), total=len(results), desc="Creating patient timelines"):
            patient_id = row['subject_id']
            if patient_id not in timelines:
                timelines[patient_id] = []
            
            event = {
                'timestamp': row['timestamp'],
                'icd_code': row['icd_code'],
                'description': row['diagnosis_description'],
                'admission_id': row['admission_id']
                
            }
            timelines[patient_id].append(event)

        return timelines
    
    def identify_cohorts(self) -> Tuple[List[str], List[str]]:
        """Split patients into diabetes and control cohorts using DuckDB."""
        diabetes_codes_str = ", ".join([f"'{code}'" for code in self.diabetes_codes])
        
        query = f"""
            WITH diabetes_patients AS (
                SELECT DISTINCT subject_id
                FROM clinical_data
                WHERE substring(CAST(icd_code AS VARCHAR), 1, 3) IN ({diabetes_codes_str})
            ),
            all_patients AS (
                SELECT DISTINCT subject_id
                FROM clinical_data
            )
            SELECT 
                a.subject_id,
                CASE WHEN d.subject_id IS NOT NULL THEN true ELSE false END as has_diabetes
            FROM all_patients a
            LEFT JOIN diabetes_patients d ON a.subject_id = d.subject_id
        """
        
        results = self.con.execute(query).fetchdf()
        
        diabetes_patients = results[results['has_diabetes']]['subject_id'].tolist()
        control_patients = results[~results['has_diabetes']]['subject_id'].tolist()
        
        return diabetes_patients, control_patients
    
    def extract_patient_features(self, patient_id: str, timelines : Dict) -> Dict:
        """Extract features for a single patient using DuckDB."""
        # Get patient demographics
        demo_query = f"""
            SELECT DISTINCT
                race,
                language,
                marital_status,
                insurance
            FROM clinical_data
            WHERE subject_id = {patient_id}
            LIMIT 1
        """
        
        demographics = self.con.execute(demo_query).fetchone()
        
        # Get admission count
        admission_query = f"""
            SELECT COUNT(DISTINCT hadm_id) as total_admissions
            FROM clinical_data
            WHERE subject_id = {patient_id}
        """
        
        total_admissions = self.con.execute(admission_query).fetchone()[0]
        
   
      
        
        features = {
            'demographics': {
                'race': demographics[0],
                'language': demographics[1],
                'marital_status': demographics[2],
                'insurance': demographics[3]
            },
            'diagnoses': timelines[patient_id],
            'total_admissions': total_admissions,
        }
        
        return features

    def prepare_all_patients_huggingface_dataset(self) -> Dataset:
        """Prepare data in format suitable for HuggingFace upload."""
        # diabetes_cohort, control_cohort = self.identify_cohorts()
        dataset_records = []
        timelines = self.create_patient_timelines() 
        for patient_id, _ in tqdm(timelines.items(), desc="Processing all patients"):
            features = self.extract_patient_features(patient_id, timelines)
            dataset_records.append({
                'patient_id': patient_id,
                'features': features
            })
        return Dataset.from_list(dataset_records)
    
 


def upload_to_huggingface(dataset: Dataset, 
                         repo_name: str,
                         private: bool = True) -> None:
    """Upload the dataset to HuggingFace Hub."""
    dataset.push_to_hub(repo_name, private=private)
    print(f"Dataset successfully uploaded to {repo_name}")

if __name__ == "__main__":
    # Initialize processor
    processor = ClinicalDataProcessor("processed_diagnoses.csv")
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = processor.prepare_all_patients_huggingface_dataset()
    
    # Upload to HuggingFace
    repo_name = "lukasnellotus/clinical-all-patients-dataset"
    
    print("Uploading to HuggingFace Hub...")
    upload_to_huggingface(
        dataset=dataset,
        repo_name=repo_name,
        private=True
    )



After grouping the patients, I identify the diabetes patients by checking to see whether any of them have the ICD-10 or ICD-9 codes for diabetes in their medical histories, and adding the new columns, “diabetes_onset” and “diagnosis_prior_to_diabetes” to those that do, as well as a “has_diabetes” column for all patients. The code for this is the following:

from datasets import load_dataset

ds = load_dataset("lukasnellotus/clinical-all-patients-dataset", split='train')
diabetes_codes = ['250', '250.0', 'E11']

def starts_with_list(string, list):
    for item in list:
        if string.startswith(item):
            return True
    return False

def get_diabetes(example) -> dict | None:
    for code in example['features']['diagnoses']:
        if starts_with_list(code['icd_code'], diabetes_codes):
            return code
    return False

def add_diabetes_label(example):
    diabetes = get_diabetes(example)
    if not diabetes:
        return {
            "has_diabetes": False,
            "diabetes_onset": None,
            "diabetes_admission_id": None,
            "diagnoses_prior_to_diabetes": []
        }
    diabetes_onset = diabetes['timestamp']
    diabetes_admission_id = diabetes['admission_id']
    diagnoses_prior_to_diabetes = [d for d in example['features']['diagnoses'] if d['admission_id'] != diabetes_admission_id and d['timestamp'] < diabetes_onset]
    return {
        "has_diabetes": True,
        "diabetes_onset": diabetes_onset,
        "diabetes_admission_id": diabetes_admission_id,
        "diagnoses_prior_to_diabetes": diagnoses_prior_to_diabetes
    }

ds = ds.map(add_diabetes_label)

ds.push_to_hub("lukasnellotus/clinical-all-patients-dataset-with-diabetes-labels", private=True)


The idea here is quite obvious: You want to only infer on conditions prior to diabetes to have a good comparison to those with no diabetes. 

After this, I do the final processing to get the actual system prompt:

from datetime import datetime
from datasets import load_dataset

ds = load_dataset("lukasnellotus/clinical-all-patients-dataset-with-diabetes-labels", split='train')

def days_between_dates(date1: datetime, date2: datetime) -> int:
    """,
    Calculate the number of days between two dates.

    """
    # Get absolute difference in days
    delta = abs((date2 - date1).days)

    return delta
def process_row(example):
      # Check if patient has enough admissions
      if example['features']['total_admissions'] < 3:
          return None

      # Get the correct diagnoses field based on diabetes status
      diagnoses = (example['diagnoses_prior_to_diabetes']
                  if example['has_diabetes']
                  else example['features']['diagnoses'])

      # Sort diagnoses by timestamp
      sorted_diagnoses = sorted(diagnoses, key=lambda x: x['timestamp'])

      # Format the patient history
      history_lines = []

      for diagnosis in sorted_diagnoses:
          first_day = sorted_diagnoses[0]['timestamp']
          # Extract year and day from timestamp (format: YYYY-MM-DDThh:mm:ss)
          days = days_between_dates(first_day,diagnosis['timestamp'])
          year, day = divmod(days, 365)

          history_line = f"{diagnosis['description']} - Year {year} Day {day}"
          history_lines.append(history_line)

      # Create the prompt
      prompt = (
          "Patient History:\n" +
          "\n".join(history_lines) +
          "\n\nBased on this medical history, will this patient develop diabetes?\n\n" +
          "Your answer should look like the following\n" +
          "<think>reasoning about why the patient has diabetes\n" +
          "</think><answer>yes</answer>\n" +
          "Please reason about and provide several reasons for why you think the patient has diabetes in the <think></think> tags. " +
          "Please provide your answer as a single diagnosis, 'yes' or 'no', in the <answer></answer> tags, with yes meaning that the patient has diabetes " +
          "and no meaning that the patient does not. Please put all the reasoning in the <think></think> tags, and don't write anything outside the <think></think> and <answer></answer> tags\n"
      )

      # Update the example with the new prompt
      example['user_prompt'] = prompt
      return example

ds = ds.filter(lambda example: example['features']['total_admissions'] >= 3, num_proc=16)
# Apply the transformation to each row
ds = ds.map(process_row, num_proc=16)
# Filter out None values (rows with < 3 admissions)
ds = ds.filter(lambda x: x is not None, num_proc=16)
ds.push_to_hub(
    'lukasnellotus/clinical-all-patients-dataset-with-diabetes-labels-and-prompt',
    private=True,
)

