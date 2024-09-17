# Session 1.1 Notes
[Notes by Peter Ernicke](https://knowmledge.com/2023/09/09/ml-zoomcamp-2023-introduction-to-machine-learning-part-1/) - Actually super helpful and pretty complete. 

The concept of ML is depicted with an example of predicting the price of a car. The ML model learns from data, represented as some features such as year, mileage, among others, and the target variable, in this case, the car's price, by extracting patterns from the data. Think of ML as a way that we can use the knowledge of an "expert" in a model. 

Then, the model is given new data (without the target, a.k.a. features) about cars and predicts their price (target).

In summary, ML is a process of extracting patterns from data, which is of two types:

    features (information about the object) and
    target (property to predict for unseen objects).

Therefore, new feature values are presented to the model, and it makes predictions from the learned patterns.


## Notes from Emmanuel: 

Just based off of the first lesson, I'm thinking it might be a good idea to do some real estate price prediction.

# Session 1.2 Notes 

 [Notes from Peter Ernicke](https://knowmledge.com/2023/09/10/ml-zoomcamp-2023-introduction-to-machine-learning-part-2/)

## Notes

The differences between ML and Rule-Based systems is explained with the example of a **spam filter**.

Traditional Rule-Based systems are based on a set of **characteristics** (keywords, email length, etc.) that identify an email as spam or not. As spam emails keep changing over time the system needs to be upgraded making the process untractable due to the complexity of code maintenance as the system grows.

ML can be used to solve this problem with the following steps:

### 1. Get data 
Emails from the user's spam folder and inbox gives examples of spam and non-spam.

### 2. Define and calculate features
Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox).

Each email can be encoded (converted) to the values of it's features and target.

### 3. Train and use the model
A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The **predictions are probabilities**, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam. 


# Session 1.3 Supervised Machine Learning 

## Notes

In Supervised Machine Learning (SML) there are always labels associated with certain features.
The model is trained, and then it can make predictions on new features. In this way, the model
is taught by certain features and targets. 

* **Feature matrix (X):** made of observations or objects (rows) and features (columns).
* **Target variable (y):** a vector with the target information we want to predict. For each row of X there's a value in y.


The model can be represented as a function **g** that takes the X matrix as a parameter and tries
to predict values as close as possible to y targets. 
The obtention of the g function is what it is called **training**.

### Types of SML problems 

* **Regression:** the output is a number (car's price)
* **Classification:** the output is a category (spam example). 
	* **Binary:** there are two categories. 
	* **Multiclass problems:** there are more than two categories. 
* **Ranking:** the output is the big scores associated with certain items. It is applied in recommender systems. 

In summary, SML is about teaching the model by showing different examples, and the goal is
to come up with a function that takes the feature matrix as a
parameter and makes predictions as close as possible to the y targets. 

# Session 1.4 CRISP-DM

[Notes from Peter Ernicke](https://knowmledge.com/2023/09/12/ml-zoomcamp-2023-introduction-to-machine-learning-part-4/)

## Notes

CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an open standard process model that describes common approaches used by data mining experts. It is the most widely-used analytics model. Was conceived in 1996 and became a European Union project under the ESPRIT funding initiative in 1997. The project was led by five companies: Integral Solutions Ltd (ISL), Teradata, Daimler AG, NCR Corporation and OHRA, an insurance company: 

1. **Business understanding:** An important question is: "If we need ML for the project?" The goal of the project has to be measurable. 
- Many projects don't need ML at all. So is ML the right solution for this?

Let's say we have the following problem: 
* Users complain about spam
* We then analyze how big the problem is
* Ask ourselves if ML is the right kind of solution -> Maybe we will be fine with developing some heuristic without spending a lot of resources 

If we are going with a machine learning model, then we need to create metrics: 
**Define the goal** 
* Reduce the amount of spam messages, or 
* Reduce the amount of complaints about spam 

Make it measurable! "Reduce amount of spam messages by 50%" 

2. **Data understanding:** Analyze available data sources, and decide if more data is required. 

**Identify data sources** 
* ID'ing data sources influences the goal - We may need to completely reconfigure our goal. 

**We have a report spam button** 
* Is the data behind this button good enough?
* Is it reliable?
* Do we track it correctly?
* Is the dataset large enough? Small data doesn't get us anywhere
* Do we need to get more data? 

3. **Data preparation:** Clean data and remove noise applying pipelines, and the data should be converted to a tabular format, so we can put it into ML.

* Usually this means extracting different features, 
* Cleaning the data,
* Building a pipeline, etc

4. **Modeling:** training Different models and choose the best one. Considering the results of this step, it is proper to decide if is required to add new features or fix data issues. 

**Which model to choose?**
* Logistic Regression -> 
* Decision Tree 
* Neural Network 

**Adding features**
Sometimes you need to go back to the data preparation step and fix up what you did before because the model could perform better with additional features. 

5. **Evaluation:** Measure how well the model is performing and if it solves the business problem. 

**Is the model good enough?**
* Have we reached the goal?
* Do our metrics improve? 

**Goal: Reduce the amount by 50%**
- Have we reduced it and by how much?
- Evaluate on the test group 

**Do a retro:**
- Was the goal even achievable? 
- Did we solve/measure the right thing? 

**Afterwards we may want to:**
- Go back and adjust the goal 
- Rolle the model to more/all users
- Stop development 

6. **Deployment:** Roll out to production to all the users. The evaluation and deployment often happen together - **online evaluation**. 

- Most times nowadays, we'll test model on real users (online evaluation)

It is important to consider how well maintainable the project is.
  
In general, ML projects require many iterations.

**Iteration:** 
* Start simple
* Learn from the feedback
* Improve