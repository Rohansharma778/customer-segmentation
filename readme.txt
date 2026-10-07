"The K-Means clustering model achieved a Silhouette Score of 0.5653, indicating reasonably well-separated and cohesive customer segments."


"I selected four business-relevant customer features: income, total spending, campaign acceptance, and interaction. I standardized the features and applied PCA to reduce dimensionality while retaining 84.21% of the variance. I evaluated K-Means for different values of K using silhouette scores and selected K=3, which achieved a silhouette score of 0.5653. I then profiled the resulting clusters based on their average customer characteristics and developed different marketing strategies for each segment."

 #####important command:   ".\.venv311\Scripts\Activate.ps1"

 path = kagglehub.dataset_download("rodsaldanha/arketing-campaign")