## March Mandess Bracket Generator

This is the public repo for my bracket generator using dummy values for data because I don't want to get sued 🙃

The website is currently being hosted at https://zkolano.com if you want to check it out

### Parameter Selection

It initially brings the user to a parameter selection screen, allowing them to pick from ~250 unique parameters sourced from various analysts' websites and personal scraping.
These are used if you want to use specific data, such as mainly offensive stats if you want to see that correlation.
The user also selects a year of past data to run these predictions on.

<img width="1190" alt="image" src="https://github.com/user-attachments/assets/6cbb8c12-d294-4d8f-927b-2bb68fd9906f" />



### Model Selection

This page essentially provides all keystone sci-kit-learn machine learning models to train on, as well as hyperparameter selection and the choice to either make it a regression or classification task.

<img width="1199" alt="image" src="https://github.com/user-attachments/assets/a54846e6-42b3-49e8-862b-c9e74815c17a" />


### Results

After training and running a model on that year's data, it brings them to a results page, showing their bracket "score" as well as other metrics on how it performed

<img width="1186" alt="image" src="https://github.com/user-attachments/assets/9e80388a-f265-4dfc-9b3e-680edc016d5e" />


### Application

But what good is a model if you can't use it for current data. Once you get to the results page, there's an option to apply that model on the current dataset, which takes in the latest scraped data and uses
that to predict either the latest Lunardi projections or the actual tournament teams depending on how far along the season is.
