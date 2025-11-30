# Project Requirements Specification (PRS)

> **Project Name**: Quantitative electroencephalography (qEEG) 
> **Version**: v1.02  
> **Date**: 2025/11/23  
> **Target Model**: Codex - GPT-5
> **Current environment** :
> Conda virtual environment: "F:\ProgramData\anaconda3\envs\mne_1.9.0\python.exe" (All the required packages have been installed.)
> System: Windows11
> Hardware: [ 6 cores / 48GB RAM / 16GB Cuda RAM ]
> IDE/Editor: Vscode
---

##  Project Overview

###  Project Objectives
**Brief Description** :
- An extensible quantitative electroencephalogram (EEG) analysis tool based on Python-MNE

**Detailed Description**:
- Electroencephalogram (EEG) data in common formats can be calculated
- Multiple metrics can be calculated (absolute power of each channel, relative power, etc.)
- Subsequent modifications can easily expand and enhance the function, allowing for the computation of more complex EEG features.
- The feature output format is standardized, facilitating statistical analysis or downstream tasks such as machine learning

---

##  Functional Requirements

###  Core Feature List
**P0 - Must Have**:
- Quality Control Report
- qEEG result outputs (long-format tidy dataset)
- Can add a new feature calculation module

**P1 - Important but Not Critical**:
- Real-time task processing progress

**P2 - Future Consideration**:
- Visualized web page

---

##  Architecture Design

###  Directory Structure
```
project-root/
├── docs/                  
│   ├── Project-Requirements-Specification.md
│   ├── Architecture.md    # Architecture file
│   ├── Planning.md          # Planning History
├── data/                  
│   ├── EEG_DATA/          # .fif or .edf files
├── utils/                 # Callable python tool script           
├── configs/               
│   ├── [function].json    
├── result                 
│   ├── [Year-Month-Day-m-s]/
│       ├── **Output**     # **Output** files in 4.1 Description
├── code[number]_[function].py  # The main execution procedure parses parameters in configs/ and calls the tools in the utils/ to implement specific functions
├── requirements.txt       # Dependencies list
├── README.md              # Project documentation
```

###  Main Execution Program
- code_[number]_[function].py # Implement Entitys

###  Functional Module
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

**Entity 4**: 
- The library based on: AntroPy
- Source code location: project-root/utils/entropy.py
> Specific functions: Calculate the Spectral Entropy of each channel of the EEG data.
>> Example code in https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html#antropy.spectral_entropy
>> Optional parameters:(method,nperseg,normalize)

**Optional parameters**: Configuration file content
- Data dir and Result dir
- I can choose different montage.
- I can choose different reference patterns.
- Power-related **Entitys** can specify different calculation frequency bands and calculation parameters.
- Entropy-related **Entitys** can specify different calculation frequency bands and calculation parameters.
---

##  Interface Specifications

###  Input/Output Definitions
**Input**:
- config file path ([function].json)

**Output**:
- Quality Control Report.html
> Create a webpage with a dropdown menu at the top that displays different feature content based on the selected dropdown option. The first option summarizes the parameters of the EEG data, including the parameters of the EEG data header. Each of the other options represents a feature and shows the feature distribution and outlier situation for each EEG data instance (including channel-level and group mean-level). All results are presented through histograms or other graphics and tables.

- qEEG_result.csv 
> long-format tidy dataset. The first column is the patient ID (EEG file name).

- log 
> logs file
---

##  Non-Functional Requirements

###  Maintainability
- Critical functions require docstring comments
- Complex logic needs inline comments

###  Extensibility
- Use dependency injection pattern
- Separate configuration from code
- Maintain loose coupling between modules

###  Observability
- Log critical operations

---

##  Acceptance Criteria

###  Functional Acceptance
- Users can create tasks via CLI

###  Deliverables Checklist
- Add your Planning to docs/Planning.md
- Update docs/Architecture.md
- Source code (with comments)
- README.md (including installation and usage instructions)
- requirements.txt

---

##  Extensibility Considerations

###  Known Extension Points
- 

###  Architecture Provisions
- The newly added quantitative calculation functions will be placed in the /utils folder, creating different sub-tool scripts within the /utils folder based on the dependencies of the libraries.

---


### A. Change Log
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| v1.0  | 2025-11-23 | Initial version | xiaoyi |
| v1.01 | 2025-11-23 | Add **Entity 3** Permutation entropy | xiaoyi |
| v1.02 | 2025-11-25 | Add **Entity 4** Spectral entropy | xiaoyi |

