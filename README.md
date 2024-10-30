# EGD_CXR Perceptual Error Generation

This repository contains code to generate perceptual errors for CT scan gaze fixation and transcript data.

# Required File Structure

- `audio_segmentation_transcripts` and `fixations` directories must be located in the root directory and contain the transcript and gaze fixation data respectively.

```bash
.
├── audio_segmentation_transcripts
│   └── {.dcm Id}
│       └── transcript.json
└── fixations
    └── {.dcm Id}
        └── fixations.csv
```

# How to use

1. Pull the repository or download the file and navigate to the root directory.
2. Run the following commands to install the required packages and run the script.

```bash
pip install -r requirements.txt

python process.py
```

# Expected Output

```bash
.
├── aggregated_dicom_data.json
├── aggregated_dicom_perceptual_error_info.json
└── dicom_abnormality_transcript_timestamps.json

```
