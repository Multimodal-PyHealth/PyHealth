"""
Mock equivalent of examples/mortality_prediction/multimodal_mimic4.py.

Runs both ClinicalNotesMIMIC4 and ClinicalNotesICDLabsMIMIC4 on synthetic
patients instead of real MIMIC-IV files, so the pipeline can be exercised
without any local data.

Mock infrastructure is imported from test_multimodal_mimic_4.py in this
directory. See that file for the interface citations.
"""
import os
import pprint
from datetime import datetime
from unittest.mock import MagicMock

from test_multimodal_mimic_4 import MockNote, MockLabEvent, MockAdmission, MockPatient
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4, ClinicalNotesICDLabsMIMIC4


class MockICDEvent:
    """Mirrors the Event interface for diagnoses_icd / procedures_icd rows.
    Field names from pyhealth/datasets/configs/mimic4_ehr.yaml."""
    def __init__(self, icd_code):
        self.icd_code = icd_code


# ---------------------------------------------------------------------------
# Mock patients
# ---------------------------------------------------------------------------

# Two admissions; patient survives both → mortality = 0
mock_survivor = MockPatient("p_survivor", admissions=[
    MockAdmission(
        hadm_id="hadm_1",
        timestamp=datetime(2023, 1, 1, 8, 0),
        dischtime="2023-01-05 14:00:00",
        # discharge_notes=[
        #     MockNote(
        #         "Patient admitted with community-acquired pneumonia. "
        #         "Treated with IV antibiotics. Condition improved.",
        #         datetime(2023, 1, 4, 10, 0),   # 74h after admission
        #     )
        # ],
        radiology_notes=[
            MockNote(
                "CXR: right lower lobe infiltrate consistent with pneumonia. "
                "No pleural effusion.",
                datetime(2023, 1, 1, 12, 0),   # 4h after admission
            )
        ],
        lab_events=[
            MockLabEvent(itemid="50983", valuenum=138.0,   # Sodium
                         timestamp=datetime(2023, 1, 1, 10, 0),
                         storetime="2023-01-01 10:00:00"),
            MockLabEvent(itemid="50971", valuenum=4.1,     # Potassium
                         timestamp=datetime(2023, 1, 1, 10, 0),
                         storetime="2023-01-01 10:00:00"),
            MockLabEvent(itemid="50902", valuenum=101.0,   # Chloride
                         timestamp=datetime(2023, 1, 1, 10, 0),
                         storetime="2023-01-01 10:00:00"),
            MockLabEvent(itemid="50983", valuenum=140.0,   # Sodium (repeat at 24h)
                         timestamp=datetime(2023, 1, 2, 8, 0),
                         storetime="2023-01-02 08:00:00"),
        ],
        diagnoses_icd=[
            MockICDEvent("J189"),    # Pneumonia, unspecified
            MockICDEvent("Z8791"),   # Personal history of pneumonia
        ],
        procedures_icd=[
            MockICDEvent("3E0936Z"), # IV antibiotic administration
        ],
    ),
    MockAdmission(
        hadm_id="hadm_2",
        timestamp=datetime(2023, 6, 1, 9, 0),
        dischtime="2023-06-04 11:00:00",
        discharge_notes=[
            MockNote(
                "Follow-up admission. Shortness of breath, resolved with bronchodilators.",
                datetime(2023, 6, 3, 15, 0),
            )
        ],
        radiology_notes=[
            MockNote(
                "CXR: lungs clear. No acute cardiopulmonary process.",
                datetime(2023, 6, 1, 11, 0),
            )
        ],
        lab_events=[
            MockLabEvent(itemid="50983", valuenum=142.0,   # Sodium
                         timestamp=datetime(2023, 6, 1, 11, 0),
                         storetime="2023-06-01 11:00:00"),
            MockLabEvent(itemid="50971", valuenum=3.9,     # Potassium
                         timestamp=datetime(2023, 6, 1, 11, 0),
                         storetime="2023-06-01 11:00:00"),
        ],
        diagnoses_icd=[
            MockICDEvent("J459"),    # Asthma, unspecified
        ],
    ),
])

# One admission with no notes at all — discharge_notes and radiology_notes
# are omitted, so the task will fill both with TOKEN_REPRESENTING_MISSING_TEXT
# and TOKEN_REPRESENTING_MISSING_FLOAT placeholders.
mock_missing_notes = MockPatient("p_missing_notes", admissions=[
    MockAdmission(
        hadm_id="hadm_1",
        timestamp=datetime(2023, 5, 1, 8, 0),
        dischtime="2023-05-04 12:00:00",
        # discharge_notes and radiology_notes intentionally omitted
        lab_events=[
            MockLabEvent(itemid="50983", valuenum=139.0,   # Sodium
                         timestamp=datetime(2023, 5, 1, 10, 0),
                         storetime="2023-05-01 10:00:00"),
        ],
        diagnoses_icd=[
            MockICDEvent("I10"),     # Essential hypertension
        ],
    ),
])

# Two admissions; second has hospital_expire_flag=1 → mortality = 1.
# Only the first admission's data is included in the sample (the task
# stops collecting once it sees the death flag on the next admission).
mock_nonsurvivior = MockPatient("p_nonsurvivior", admissions=[
    MockAdmission(
        hadm_id="hadm_1",
        timestamp=datetime(2023, 3, 1, 6, 0),
        dischtime="2023-03-07 18:00:00",
        discharge_notes=[
            MockNote(
                "Patient admitted with sepsis secondary to UTI. "
                "Started on broad-spectrum antibiotics.",
                datetime(2023, 3, 6, 9, 0),
            )
        ],
        radiology_notes=[
            MockNote(
                "CXR: mild pulmonary vascular congestion. No focal consolidation.",
                datetime(2023, 3, 1, 8, 0),
            )
        ],
        lab_events=[
            MockLabEvent(itemid="50983", valuenum=132.0,   # Sodium (low)
                         timestamp=datetime(2023, 3, 1, 7, 0),
                         storetime="2023-03-01 07:00:00"),
            MockLabEvent(itemid="50971", valuenum=5.2,     # Potassium (high)
                         timestamp=datetime(2023, 3, 1, 7, 0),
                         storetime="2023-03-01 07:00:00"),
            MockLabEvent(itemid="50868", valuenum=18.0,    # Anion Gap (elevated)
                         timestamp=datetime(2023, 3, 1, 7, 0),
                         storetime="2023-03-01 07:00:00"),
        ],
        diagnoses_icd=[
            MockICDEvent("A419"),    # Sepsis, unspecified
            MockICDEvent("N390"),    # UTI
        ],
    ),
    MockAdmission(
        hadm_id="hadm_2",
        timestamp=datetime(2023, 3, 15, 2, 0),
        dischtime="2023-03-20 00:00:00",
        hospital_expire_flag=1,   # patient dies in this admission
    ),
])


# ---------------------------------------------------------------------------
# Run tasks (mirrors dataset.set_task() but over mock patients)
# ---------------------------------------------------------------------------
TASK = "ClinicalNotesICDLabsMIMIC4"

if __name__ == "__main__":
    # patients = [mock_survivor, mock_nonsurvivior]
    patient = mock_survivor
    task = ClinicalNotesICDLabsMIMIC4()

    samples = task(patient)
    print(f"\n{'='*60}")
    print(f"Patient: {patient.patient_id}  (task: {TASK})")
    print('='*60)
    for sample in samples:
        pprint.pprint(sample)
