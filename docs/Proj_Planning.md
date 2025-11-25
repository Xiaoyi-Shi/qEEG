# Project Requirements Specification (PRS)

> **Project Name**: Quantitative electroencephalography (qEEG) 
> **Version**: v1.01  
> **Date**: 2025/11/23  
> **Target Model**: Codex - GPT-5
> **Current environment** :
> Conda virtual environment: mne_1.9.0 (All the required packages have been installed.)
> System: Windows11
> Hardware: [ 6 cores / 48GB RAM / 16GB Cuda RAM ]
> IDE/Editor: Vscode
---

## 1. Project Overview

### 1.1 Project Objectives
**Brief Description** :
- An extensible quantitative electroencephalogram (EEG) analysis tool based on Python-MNE

**Detailed Description**:
- Electroencephalogram (EEG) data in common formats can be calculated
- Multiple metrics can be calculated (absolute power of each channel, relative power, etc.)
- Subsequent modifications can easily expand and enhance the function, allowing for the computation of more complex EEG features.
- The feature output format is standardized, facilitating statistical analysis or downstream tasks such as machine learning

---

## 2. Functional Requirements

### 2.1 Core Feature List
**P0 - Must Have**:
- Quality Control Report
- qEEG result outputs (long-format tidy dataset)
- Can add a new feature calculation module

**P1 - Important but Not Critical**:
- Real-time task processing progress

**P2 - Future Consideration**:
- Visualized web page

---

## 3. Architecture Design

### 3.1 Directory Structure
```
project-root/
├── docs/                  
│   ├── Proj_Planning.md   # PRS file
│   ├── CHANGELOG.md       # PRS change log
│   ├── Architecture.md    # Architecture Document
│   ├── Prompt.md          # Prompt history
├── data/                  
│   ├── EEG_DATA/          # .fif or .edf
├── utils/                 
│   ├── xxxx.py            # Callable python tool script
├── configs/               
│   ├── cal_qEEG_all.json  # **Input** files in 4.1 Description
├── result                 
│   ├── Year-Month-Day-m-s/
│   ├── **Output**         # **Output** files in 4.1 Description
├── code_01_xxxx.py        # python script that can accomplish a certain task
├── requirements.txt       # Dependencies list
├── README.md              # Project documentation
```

### 3.2 Functional Module
**Entity 1**: Absolute power(μV^2/Hz):
- The library based on: MNE
- Source code location: project-root/utils/basefun.py
- use raw.compute_psd() function to compute
> Specific functions: Calculate the absolute power of each channel of the EEG data.
>> The frequency band and other parameters of the calculation can be adjusted through the configuration file.


**Entity 2**: Relative power(Ratio):
- The library based on: MNE
- Source code location: project-root/utils/basefun.py
> Specific functions: Calculate the relative power of each channel of the EEG data.
>> The frequency band and other parameters of the calculation can be adjusted through the configuration file.

**Entity 3**: 
- The library based on: AntroPy
- Source code location: project-root/utils/entropy.py
> Specific functions: Calculate the Permutation entropy (Customize special frequency bands in the config) of each channel of the EEG data.
>> Example
>> ``` 
>> import numpy as np
>> import antropy as ant
>> np.random.seed(1234567)
>> x = np.random.normal(size=3000)
>> print(ant.perm_entropy(x, normalize=True))
>> ```

---

## 4. Interface Specifications

### 4.1 Input/Output Definitions
**Input**:
- config file path (xxxx.json)
> - EEG data dir/file path (.fif/.edf) 
> - Parameters needed to calculate various features.

**Output**:
- Quality Control Report.html
> Create a webpage with a dropdown menu at the top that displays different feature content based on the selected dropdown option. The first option summarizes the parameters of the EEG data, including the parameters of the EEG data header. Each of the other options represents a feature and shows the feature distribution and outlier situation for each EEG data instance (including channel-level and group mean-level). All results are presented through histograms or other graphics and tables.

- qEEG_result.csv 
> long-format tidy dataset. The first column is the patient ID (EEG file name).

- log 
> logs file
---

## 5. Non-Functional Requirements

### 5.1 Maintainability
- Critical functions require docstring comments
- Complex logic needs inline comments

### 5.2 Extensibility
- Use dependency injection pattern
- Separate configuration from code
- Maintain loose coupling between modules

### 5.3 Observability
- Log critical operations

---

## 6. Acceptance Criteria

### 6.1 Functional Acceptance
- Users can create tasks via CLI

### 6.2 Deliverables Checklist
- Update CHANGELOG.md (run `git diff HEAD docs/Proj_Planning.md` to get changes)
- Update Architecture.md
- Source code (with comments)
- README.md (including installation and usage instructions)
- requirements.txt / package.json

---

## 7. Extensibility Considerations

### 7.1 Known Extension Points
- The configuration file can be modified to adjust parameters.

### 7.2 Architecture Provisions
- The newly added quantitative calculation functions will be placed in the /utils folder, creating different sub-tool scripts within the /utils folder based on the dependencies of the libraries.

---


### A. Change Log
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| v1.0  | 2025-11-23 | Initial version | only power cal |
| v1.01 | 2025-11-23 | Initial version | Add **Entity 3** Permutation entropy |


