import gzip
import csv
import duckdb
from typing import Generator, Dict, Any
from tqdm import tqdm

class DiagnosesLoader:
    def __init__(self, diagnoses_path: str, definitions_path: str, admissions_path: str):
        """
        Initialize the loader with file paths.
        
        Args:
            diagnoses_path (str): Path to the gzipped CSV file containing diagnoses data
            definitions_path (str): Path to the gzipped CSV file containing ICD definitions
            admissions_path (str): Path to the CSV file containing admissions data
        """
        self.diagnoses_path = diagnoses_path
        self.definitions_path = definitions_path
        self.admissions_path = admissions_path
        self.conn = None
    
    def setup_database(self):
        """Set up the database tables and load data."""
        self.conn = duckdb.connect(database=':memory:', read_only=False)
        
        # Create diagnoses table with VARCHAR for all fields initially
        self.conn.execute("""
            CREATE TABLE diagnoses (
                subject_id VARCHAR,
                hadm_id VARCHAR,
                seq_num VARCHAR,
                icd_code VARCHAR,
                icd_version VARCHAR
            )
        """)
        
        # Create ICD definitions table with VARCHAR for all fields
        self.conn.execute("""
            CREATE TABLE icd_definitions (
                icd_code VARCHAR,
                icd_version VARCHAR,
                long_title VARCHAR
            )
        """)
        
        # Create admissions table
        self.conn.execute("""
            CREATE TABLE admissions (
                subject_id VARCHAR,
                hadm_id VARCHAR,
                admittime TIMESTAMP,
                dischtime TIMESTAMP,
                deathtime TIMESTAMP,
                admission_type VARCHAR,
                admit_provider_id VARCHAR,
                admission_location VARCHAR,
                discharge_location VARCHAR,
                insurance VARCHAR,
                language VARCHAR,
                marital_status VARCHAR,
                race VARCHAR,
                edregtime TIMESTAMP,
                edouttime TIMESTAMP,
                hospital_expire_flag VARCHAR
            )
        """)
        
        # Load diagnoses with specific CSV settings
        self.conn.execute(f"""
            COPY diagnoses FROM '{self.diagnoses_path}'
            (
                FORMAT CSV,
                AUTO_DETECT FALSE,
                HEADER TRUE,
                DELIMITER ',',
                QUOTE '"',
                ESCAPE '"',
                STRICT_MODE FALSE
            );
        """)
        
        # Load ICD definitions with specific CSV settings
        self.conn.execute(f"""
            COPY icd_definitions FROM '{self.definitions_path}'
            (
                FORMAT CSV,
                AUTO_DETECT FALSE,
                HEADER TRUE,
                DELIMITER ',',
                QUOTE '"',
                ESCAPE '"',
                STRICT_MODE FALSE
            );
        """)
        
        # Load admissions with specific CSV settings
        self.conn.execute(f"""
            COPY admissions FROM '{self.admissions_path}'
            (
                FORMAT CSV,
                AUTO_DETECT FALSE,
                HEADER TRUE,
                DELIMITER ',',
                QUOTE '"',
                ESCAPE '"',
                STRICT_MODE FALSE
            );
        """)
    
    def get_joined_query(self) -> str:
        """Return the SQL query for joining the tables."""
        return """
            SELECT 
                d.subject_id,
                d.hadm_id,
                d.seq_num,
                d.icd_code,
                d.icd_version,
                COALESCE(i.long_title, 'Unknown') as diagnosis_description,
                a.admittime,
                a.dischtime,
                a.deathtime,
                a.admission_type,
                a.admit_provider_id,
                a.admission_location,
                a.discharge_location,
                a.insurance,
                a.language,
                a.marital_status,
                a.race,
                a.edregtime,
                a.edouttime,
                a.hospital_expire_flag
            FROM diagnoses d
            LEFT JOIN icd_definitions i 
                ON d.icd_code = i.icd_code 
                AND d.icd_version = i.icd_version
            LEFT JOIN admissions a
                ON d.subject_id = a.subject_id
                AND d.hadm_id = a.hadm_id
            ORDER BY d.subject_id, d.hadm_id, d.seq_num
        """
    
    def save_to_csv(self, output_path: str, batch_size: int = 10000):
        """
        Save the joined data to a CSV file with progress tracking.
        
        Args:
            output_path (str): Path where to save the CSV file
            batch_size (int): Number of rows to process at once
        """
        try:
            self.setup_database()
            
            # Get total count for progress bar
            count_result = self.conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()
            total_rows = count_result[0]
            
            # Get column names
            result = self.conn.execute(self.get_joined_query() + " LIMIT 0")
            columns = [desc[0] for desc in result.description]
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                # Process in batches with progress bar
                with tqdm(total=total_rows, desc="Exporting data") as pbar:
                    for offset in range(0, total_rows, batch_size):
                        batch_query = f"{self.get_joined_query()} OFFSET {offset} LIMIT {batch_size}"
                        result = self.conn.execute(batch_query)
                        
                        for row in result.fetchall():
                            writer.writerow(dict(zip(columns, row)))
                        
                        pbar.update(min(batch_size, total_rows - offset))
        
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None

def process_diagnoses_data(diagnoses_path: str, definitions_path: str, admissions_path: str, output_path: str):
    """
    Process the diagnoses data and save to a CSV file.
    
    Args:
        diagnoses_path (str): Path to the gzipped CSV file containing diagnoses data
        definitions_path (str): Path to the gzipped CSV file containing ICD definitions
        admissions_path (str): Path to the CSV file containing admissions data
        output_path (str): Path where to save the output CSV file
    """
    loader = DiagnosesLoader(diagnoses_path, definitions_path, admissions_path)
    loader.save_to_csv(output_path)

# Example usage:
if __name__ == "__main__":
    diagnoses_path = "diagnoses_icd.csv"
    definitions_path = "d_icd_diagnoses.csv"
    admissions_path = "admissions.csv"
    output_path = "processed_diagnoses.csv"
    
    process_diagnoses_data(diagnoses_path, definitions_path, admissions_path, output_path)
