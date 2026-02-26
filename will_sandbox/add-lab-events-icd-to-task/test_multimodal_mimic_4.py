from datetime import timedelta, datetime
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import unittest

import polars as pl

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4, ClinicalNotesICDLabsMIMIC4

# These mock classes replicate the interfaces defined in pyhealth/data/data.py
# (Patient, lines 117-236; Event.__getattr__, lines 98-114). Field names come
# from the MIMIC-IV config YAMLs:
#   - pyhealth/datasets/configs/mimic4_note.yaml  (text, hadm_id)
#   - pyhealth/datasets/configs/mimic4_ehr.yaml   (hadm_id, hospital_expire_flag,
#                                                   dischtime, itemid, valuenum,
#                                                   storetime, icd_code)
# If a general-purpose patient mock proves useful beyond this test, it should
# live closer to pyhealth/data/data.py rather than here.

class MockNote:
    """Mirrors the Event interface for discharge/radiology note rows.
    Field names from pyhealth/datasets/configs/mimic4_note.yaml."""
    def __init__(self, text, timestamp):
        self.text = text
        self.timestamp = timestamp


class MockLabEvent:
    """Mirrors the Event interface for labevents rows.
    Field names from pyhealth/datasets/configs/mimic4_ehr.yaml."""
    def __init__(self, itemid, valuenum, timestamp, storetime):
        self.itemid = itemid          # str itemid, e.g. "50983" for Sodium
        self.valuenum = valuenum      # numeric lab value
        self.timestamp = timestamp    # datetime of the measurement
        self.storetime = storetime    # str in "%Y-%m-%d %H:%M:%S" format


class MockAdmission:
    """Mirrors the Event interface for admissions rows.
    Field names from pyhealth/datasets/configs/mimic4_ehr.yaml."""
    def __init__(self, hadm_id, timestamp, hospital_expire_flag=0,
                 discharge_notes=None, radiology_notes=None,
                 dischtime=None, lab_events=None,
                 diagnoses_icd=None, procedures_icd=None):
        self.hadm_id = hadm_id
        self.timestamp = timestamp
        self.hospital_expire_flag = hospital_expire_flag
        self.discharge_notes = discharge_notes or []
        self.radiology_notes = radiology_notes or []
        self.dischtime = dischtime        # str "%Y-%m-%d %H:%M:%S", required by ClinicalNotesICDLabsMIMIC4
        self.lab_events = lab_events or []
        self.diagnoses_icd = diagnoses_icd or []
        self.procedures_icd = procedures_icd or []


class MockPatient:
    """Mirrors the Patient.get_events interface from pyhealth/data/data.py (lines 173-236)."""
    def __init__(self, patient_id, admissions):
        self.patient_id = patient_id
        self._admissions = admissions
        self._demo = MagicMock()

    def get_events(self, event_type=None, filters=None, **kwargs):
        if event_type == "patients":
            return [self._demo]
        if event_type == "admissions":
            return self._admissions
        if event_type == "labevents":
            start = kwargs.get("start")
            end = kwargs.get("end")
            return_df = kwargs.get("return_df", False)
            all_labs = [lab for adm in self._admissions for lab in adm.lab_events]
            if start:
                all_labs = [l for l in all_labs if l.timestamp >= start]
            if end:
                all_labs = [l for l in all_labs if l.timestamp <= end]
            if return_df:
                if not all_labs:
                    return pl.DataFrame(schema={
                        "timestamp": pl.Datetime,
                        "labevents/itemid": pl.Utf8,
                        "labevents/storetime": pl.Utf8,
                        "labevents/valuenum": pl.Float64,
                    })
                return pl.DataFrame({
                    "timestamp": [l.timestamp for l in all_labs],
                    "labevents/itemid": [l.itemid for l in all_labs],
                    "labevents/storetime": [l.storetime for l in all_labs],
                    "labevents/valuenum": [float(l.valuenum) for l in all_labs],
                })
            return all_labs
        if filters:
            hadm_id = next((f[2] for f in filters if f[0] == "hadm_id"), None)
            admission = next((a for a in self._admissions if a.hadm_id == hadm_id), None)
            if admission:
                if event_type == "discharge":
                    return admission.discharge_notes
                if event_type == "radiology":
                    return admission.radiology_notes
                if event_type == "diagnoses_icd":
                    return admission.diagnoses_icd
                if event_type == "procedures_icd":
                    return admission.procedures_icd
        return []

class TestClinicalNotesICDLabsMIMIC4(unittest.TestCase):
    # Sodium itemid "50983" → index 0 in LAB_CATEGORY_NAMES
    # Potassium itemid "50971" → index 1 in LAB_CATEGORY_NAMES
    mock_sodium_lab = MockLabEvent(
        itemid="50983", valuenum=140.0,
        timestamp=datetime(2023, 1, 1, 12, 0),  # 2h after admission
        storetime="2023-01-01 12:00:00",
    )
    mock_potassium_lab = MockLabEvent(
        itemid="50971", valuenum=4.2,
        timestamp=datetime(2023, 1, 1, 12, 0),  # same draw as sodium
        storetime="2023-01-01 12:00:00",
    )
    mock_late_sodium_lab = MockLabEvent(
        itemid="50983", valuenum=138.0,
        timestamp=datetime(2023, 1, 1, 14, 0),  # 4h after admission (second timepoint)
        storetime="2023-01-01 14:00:00",
    )

    patient_no_labs = MockPatient("p_no_labs", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                      dischtime="2023-01-03 10:00:00"),
    ])

    patient_with_labs = MockPatient("p_labs", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                      dischtime="2023-01-03 10:00:00",
                      discharge_notes=[MockNote("Patient stable.", datetime(2023, 1, 2, 10, 0))],
                      lab_events=[mock_sodium_lab, mock_potassium_lab]),
    ])

    patient_with_multiple_lab_timepoints = MockPatient("p_multi_labs", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                      dischtime="2023-01-03 10:00:00",
                      discharge_notes=[MockNote("Patient stable.", datetime(2023, 1, 2, 10, 0))],
                      lab_events=[mock_sodium_lab, mock_potassium_lab, mock_late_sodium_lab]),
    ])

    def test_no_lab_events_produces_missing_tokens(self):
        """When a patient has no lab events, labs output should have one entry
        of all-zero vectors (missing float tokens) at time offset 0.0."""
        samples = ClinicalNotesICDLabsMIMIC4()(self.patient_no_labs)

        self.assertEqual(len(samples), 1)
        lab_times, lab_values = samples[0]["labs"]

        self.assertEqual(len(lab_times), 1)
        self.assertEqual(lab_times[0], ClinicalNotesICDLabsMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(len(lab_values[0]), len(ClinicalNotesICDLabsMIMIC4.LAB_CATEGORY_NAMES))
        self.assertTrue(all(
            v == ClinicalNotesICDLabsMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT
            for v in lab_values[0]
        ))

    def test_lab_values_correct_vector(self):
        """Sodium (index 0) and Potassium (index 1) labs drawn at the same timestamp
        should produce a single 10-d vector with the correct values; all other
        categories should be the missing float token."""
        samples = ClinicalNotesICDLabsMIMIC4()(self.patient_with_labs)

        self.assertEqual(len(samples), 1)
        lab_times, lab_values = samples[0]["labs"]

        self.assertEqual(len(lab_times), 1)
        lab_vector = lab_values[0]

        sodium_idx = ClinicalNotesICDLabsMIMIC4.LAB_CATEGORY_NAMES.index("Sodium")
        potassium_idx = ClinicalNotesICDLabsMIMIC4.LAB_CATEGORY_NAMES.index("Potassium")

        self.assertAlmostEqual(lab_vector[sodium_idx], 140.0)
        self.assertAlmostEqual(lab_vector[potassium_idx], 4.2)
        for i, v in enumerate(lab_vector):
            if i not in (sodium_idx, potassium_idx):
                self.assertAlmostEqual(v, 0.0)

    def test_lab_time_offset(self):
        """Lab time offset should be hours elapsed since admission start."""
        samples = ClinicalNotesICDLabsMIMIC4()(self.patient_with_labs)

        lab_times, _ = samples[0]["labs"]
        # Lab drawn at 12:00, admission started at 10:00 → 2.0 hours
        self.assertAlmostEqual(lab_times[0], 2.0)

    def test_multiple_lab_timepoints(self):
        """Two distinct draw timestamps should produce two separate lab vectors
        in chronological order with correct time offsets."""
        samples = ClinicalNotesICDLabsMIMIC4()(self.patient_with_multiple_lab_timepoints)

        self.assertEqual(len(samples), 1)
        lab_times, lab_values = samples[0]["labs"]

        self.assertEqual(len(lab_times), 2)

        sodium_idx = ClinicalNotesICDLabsMIMIC4.LAB_CATEGORY_NAMES.index("Sodium")
        potassium_idx = ClinicalNotesICDLabsMIMIC4.LAB_CATEGORY_NAMES.index("Potassium")

        # First timepoint at 2h: sodium=140.0, potassium=4.2
        self.assertAlmostEqual(lab_times[0], 2.0)
        self.assertAlmostEqual(lab_values[0][sodium_idx], 140.0)
        self.assertAlmostEqual(lab_values[0][potassium_idx], 4.2)

        # Second timepoint at 4h: sodium=138.0, potassium missing (0.0)
        self.assertAlmostEqual(lab_times[1], 4.0)
        self.assertAlmostEqual(lab_values[1][sodium_idx], 138.0)
        self.assertAlmostEqual(lab_values[1][potassium_idx], 0.0)

class TestClinicalNotesMIMIC4(unittest.TestCase):

    mock_discharge_note = MockNote("Patient had pneumonia.", datetime(2023, 1, 2, 10, 0))   # 24h after hadm_1
    mock_discharge_note_2 = MockNote("Follow-up: resolving pneumonia.", datetime(2023, 1, 3, 10, 0))  # 48h after hadm_1
    mock_radiology_note = MockNote("CXR shows infiltrate.", datetime(2023, 1, 1, 14, 0))   # 4h after hadm_1

    patient_no_notes = MockPatient("p1", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0)),
        MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),
    ])

    patient_with_notes_in_first_admisssion_but_not_the_second = MockPatient("p1", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                    discharge_notes=[mock_discharge_note],
                    radiology_notes=[mock_radiology_note]),
        MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),  # no notes
    ])

    patient_with_multiple_discharge_notes_in_first_admission = MockPatient("p1", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                    discharge_notes=[mock_discharge_note, mock_discharge_note_2],
                    radiology_notes=[mock_radiology_note]),
        MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),  # no notes
    ])

    def test_both_admission_missing_notes(self):
        """When a patient has two admissions with no notes for either,
        each admission should produce the missing text and float placeholders
        defined by TOKEN_REPRESENTING_MISSING_TEXT and TOKEN_REPRESENTING_MISSING_FLOAT."""
        samples = ClinicalNotesMIMIC4()(self.patient_no_notes)

        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_1_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

        self.assertEqual(hadm_1_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_1_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

    def test_first_admission_has_notes_second_does_not(self):
        """When a patient has two admissions and only the first has notes,
        the first admission should produce real note text and time offsets in hours,
        while the second should produce the missing text and float placeholders."""
        samples = ClinicalNotesMIMIC4()(self.patient_with_notes_in_first_admisssion_but_not_the_second)

        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge, "Patient had pneumonia.")
        self.assertAlmostEqual(hadm_1_discharge_time, 24.0)
        self.assertEqual(hadm_1_radiology, "CXR shows infiltrate.")
        self.assertAlmostEqual(hadm_1_radiology_time, 4.0)

        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

    def test_multiple_discharge_notes_per_admission(self):
        """hadm_1 has 2 discharge notes (at 24h and 48h) and 1 radiology note (at 4h).
        hadm_2 has no notes. Verifies that both discharge notes from hadm_1 are collected
        in order with correct time offsets, and that hadm_2 gets the missing text and float
        tokens for both discharge and radiology."""
        samples = ClinicalNotesMIMIC4()(self.patient_with_multiple_discharge_notes_in_first_admission)

        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge_1, hadm_1_discharge_2, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time_1, hadm_1_discharge_time_2, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge_1, "Patient had pneumonia.")
        self.assertAlmostEqual(hadm_1_discharge_time_1, 24.0)
        self.assertEqual(hadm_1_discharge_2, "Follow-up: resolving pneumonia.")
        self.assertAlmostEqual(hadm_1_discharge_time_2, 48.0)

        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

        self.assertEqual(hadm_1_radiology, "CXR shows infiltrate.")
        self.assertAlmostEqual(hadm_1_radiology_time, 4.0)
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

    def test_task_schema(self):
        """Verify that ClinicalNotesMIMIC4 exposes the expected class-level schema attributes
        with the correct keys for inputs (discharge and radiology note times) and output (mortality)."""
        self.assertIn("task_name", vars(ClinicalNotesMIMIC4))
        self.assertIn("input_schema", vars(ClinicalNotesMIMIC4))
        self.assertIn("output_schema", vars(ClinicalNotesMIMIC4))

        self.assertEqual("ClinicalNotesMIMIC4", ClinicalNotesMIMIC4.task_name)

        self.assertIn("discharge_note_times", ClinicalNotesMIMIC4.input_schema)
        self.assertIn("radiology_note_times", ClinicalNotesMIMIC4.input_schema)
        self.assertIn("mortality", ClinicalNotesMIMIC4.output_schema)
