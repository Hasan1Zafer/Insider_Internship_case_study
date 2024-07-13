# Insider_Internship_case_study
Hasan Zafer Bilir İnsider case study work

Here is limk for jupyter Notebook:
https://jupyter.org/try-jupyter/notebooks/index.html?path=notebooks%2Fml_insider_hasanzaferbilir.ipynb 

Documentation-

-Building the Docker Image:
To build the Docker image, xnavigate to the directory containing the Dockerfile and run the following command:

-- docker build -t product-category-predictor .

-Running the Docker Container
To run the Docker container, use the following command:

-- docker run -p 5000:5000 product-category-predictor

Using the Inference Interface
Making Predictions

To make predictions using the inference interface, you can send POST requests to the /predict endpoint with a product description. Below are examples using curl and Postman.
You can use the curl command-line tool to make predictions. Run the following command:

--  curl -X POST -H "Content-Type: application/json" -d '{"description": "Your product description here"}' http://localhost:5000/predict
