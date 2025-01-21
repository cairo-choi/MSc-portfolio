# A 100$ JOURNEY: unveiling Bitcoin's Growth and Public Sentiment
For the project assignment, I integrated the coursework from both Cloud Technologies and Data Management and Visualization into a single comprehensive project. In the Cloud Technologies component, I focused on AWS Cloud Computing, leveraging an EMR (Elastic MapReduce) environment with three nodes to process data using PySpark. Regarding distributed storage system, I utilized Amazon S3 because it integrates seamlessly within the AWS ecosystem, offering great convenience.

## Abstract
Our assignment aims to reveal the law of Bitcoin price trends through comparative analysis and sentiment analysis. As a digital asset with high volatility and market influence, Bitcoin's price is driven by a variety of factors, including macroeconomic changes, policy dynamics, and public sentiment. Through data collection and analysis, we will explore the relationship between social sentiment and price fluctuations, and reveal how emotional changes affect Bitcoin's market performance.
At the same time, the study will reveal historical ups and downs data and sentiment trends respectively to find the correlation between Bitcoin price factors. The results surprised us. The price of Bitcoin has fluctuated significantly in the past ten years, and it is positively correlated with people's positive comments.

## Data Collecting, Cleaning and Processing 
To emphasize the price changes of Bitcoin, we obtained the data of 9 major stocks(Apple, Telsa, Tencent, Alibaba, BYD, MaoTai, Microsoft, Netflix, Amazon) using the Yahoo Finance API. The source code for data gathering and processing can be found in the file: data_gathering_processing_pandas.ipynb

Additionally, to understand public sentiment toward Bitcoin, we performed sentiment analysis using BERT. The data for this analysis was sourced from https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets, and we extend our gratitude to Kash for sharing it. The source code for data processing using PySpark and BERT implementation can be found in the file: pyspark_emr_bert.ipynb

## AWS configuration
The most challenging part in the initial phase of the project was the AWS configuration process. I set up three nodes in the EMR environment: a Main Node, a Core Node, and a Worker Node, all configured as m5.xlarge instances. 

The key point here is that PySpark processes data in parallel across different nodes. Therefore, it is essential to ensure that the required BERT model and other necessary packages for data processing are downloaded on all three nodes.

Method 1: Use a script on the master node to simutaneously install the required packages on the core and worker nodes.

Method 2: Manually log into each of the three nodes and download the packages individually.

Since the number of nodes is small, I opted for Method 2. The commands for downloading the necessary packages are as follows:

sudo python3 -m pip install numpy pandas findspark transformers torch vaderSentiment textblob


## Data Visualization
For details on the data visualization methods and final results, please refer to the Data Management and Visualization directory.

## Presentation
You can find a detailed presentation at the link below.

https://drive.google.com/file/d/100CMJwDsjSgNHao85esYtwhrWdnuXspw/view?usp=sharing
 
