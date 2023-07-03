#Package Versions:
google-api-core==2.11.0
google-auth==2.17.2
google-cloud-core==2.3.2
google-cloud-storage==2.8.0
google-cloud-videointelligence==2.11.1
scikit-image==0.18.1
summa==1.2.0
yake==0.4.8

#Libraries to install:
!pip install --upgrade google-cloud-videointelligence
!pip install --upgrade google-cloud-storage
!pip install yake
!pip install summa

#Initial Setup: 
1. Create a bucket and on GCP, and store video data in that GCP bucket. 
2. Enable Video Intelligence API on GCP. 
3. Create a service account with required permissions for accessing the video data from bucket. Also, create a key for service account in json format which basically will allow us to use Google Video Intelligence API in our local environment.
4. Download and save the key file in your working directory where the source code will be present.

#Input to be taken from user after execution of source code:
1. User need to provide key file name when it is asked. (For example; Please enter the key file in json format: video-api-service-account.json)
2. User need to provide the video data URI of GCP bucket when it is asked. (For example; Please enter the GCP bucket URI of video: gs://ba-demo-bucket/AdCat tranche v2/1-BHP-MicroCopper-MoodMatic.mp4)

#Major Functions in source code:
1. key_path(): In this function we are taking key file name in json format from the user as input.
2. detect_logo(): This function takes GCP URI path of the video as parameter and returns logo of the video.
3. speech_to_txt(): This function takes GCP URI path of the video as parameter and returns transcript of the video.
4. logo_speech(): In this function we are calling the above two functions (detect_logo() & speech_to_txt()) and returns the logo as well as transcript of the video.
5. cleaning_text(): This function will clean the extracted video transcript and returns the cleaned text.
6. yake_extract(): This function will extract and returns the dominant words from the cleaned text using Yake package. 
7. summa_extract(): This function also will extract and returns the dominant words from the cleaned text using Summa package.
8. clean_to_dom_words(): In this function we are calling the above three functions (cleaning_text(), yake_extract() and summa_extract()) and it returns the combined dominant words of Yake and Summa packages.
9. model_building(): In this function we are training three classification models (Multinomial Naive Bayes, Decision Tree and Random Forest) using dataset "company_df".
10. test_model(): In this function we are predicting industry classification of a video using the best classification model.


#Prerequisite files to store in the source code working directory:
1. key json file
2. Training dataset which is "company_df"

#Steps for new user
1. Need to import marketing file.
2. Provide json key file name as a user input.
3. Provide the video data URI of GCP bucket as a user input. 
4. The industry classification will be given as a result.






