# Sentiment-Analysis-of-Movie-Reviews
Here I use a recurrent neural network to work out whether a movie review is positive or negative

This is my first project using a recurrent neural network, working from the tensorflow website on the starter project. I use a dataset full of worded movie reviews from IMDB, and their labels of whether they are positive or negative

I used a sequential model, with an embedding layer of dimensions 32. This turned each word into a vector of shape 32, and then used an lstm layer to remember past data and work out whether the review was positive or negative. 

It had an accuracy of 83%, and worked with my testing. 
