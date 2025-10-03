1- Création de particules
2- Analyse Formes
    a- shape différence
    b- LDA
    Anomaly detection
          c- error reconstruction
          d- LDA train test
          e- Classicifation

# Tutorial: Generating Correspondence Points with Python

This repository contains a Python script to automatically generate **correspondence points** from a Label Map database.  

With a single script execution, you will obtain:  
- an **output folder** containing the correspondence points,  
- a **ZIP file** with all the results.  

---

##  Usage

### Configure the script  
In the `pipeline_2.py` file, set the following:  
- **Project name**  
- **Path to the Label Map database**
  
<img width="622" height="115" alt="image" src="https://github.com/user-attachments/assets/8885eb6e-ab71-4e82-b7e6-7e4592d0a86a" />

You can configure the path of the ZIP output folder at the end.
<img width="652" height="146" alt="image" src="https://github.com/user-attachments/assets/c907cb4b-cd05-4b4b-b9dc-39846bedfebf" />


---

### Define optimization parameters  
Adjust the **ShapeWorks optimization parameters** (iterations, regularization, etc.) inside the script.  

📖 See the full documentation here: [ShapeWorks Parameters](https://sciinstitute.github.io/ShapeWorks/latest/parameter-files/)  

<img width="622" height="693" alt="image" src="https://github.com/user-attachments/assets/10c2b1ab-4189-466f-b8f4-75724ba266b1" />


---

### Run the script  
Run the script from the terminal:  

```bash
python pipelin_2.py






