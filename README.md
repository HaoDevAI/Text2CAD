# Thesis Project: Bilingual Text-to-CAD Model Generation (English-Vietnamese)

This project provides a pipeline and web application for generating 3D CAD models from bilingual (English and Vietnamese) text descriptions. It includes:

- Data processing and reasoning notebooks for preparing and evaluating datasets.
- CAD files and sample outputs for benchmarking and testing.
- Model fine-tuning configuration and training environment
- A quantization module for model optimization.
- A web application (frontend and backend) for user interaction and model serving.

## Main Features
- Support for both English and Vietnamese text inputs.
- Automated evaluation and metrics for generated CAD models.
- Modular design for easy extension and integration.

## Structure
- `data_processing/`: Data processing, evaluation, and metrics notebooks.
- `model_fine_tuning/`: Model fine-tuning configuration.
- `output/`, `temp/`, `models/`: Results, temporary files, and model storage.
- `quantize/`: Scripts for model quantization and conversion.
- `webapp/`: Full-stack web application (Next.js frontend, Python backend).

## Acknowledgement
We would like to sincerely thank our supervisor, Mr. Nguyen Quoc Trung, for his guidance, helpful
feedback, and encouragement during this project. His knowledge and advice greatly supported
us in shaping the direction and improving the quality of our work. We are honored to
acknowledge the support of Pythera AI for facilitating access to hardware training resources and
for providing valuable advice and guidance. We also want to thank all members of Group
GSU25AI04 for their teamwork, dedication, and effort, which made this project possible.

## Contact
For any questions or suggestions, please contact:
- Huynh Phuoc Truong Sinh: [truongsinh.work@gmail.com] (Project Leader)
- Nguyen Anh Hao: [contact.haonguyen@gmail.com]
