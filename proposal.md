
# Github User Segmentation project proposal

## Motivation and purpose
### Target Audience:
> The target audience for a GitHub user segmentation dashboard is be developers, project managers, and stakeholders who are interested in gaining insights into the usage patterns and behavior of GitHub users.
More specifically, the dashboard may be useful for teams working on open-source projects, enterprise software development, or collaborative coding projects. It can help them understand how users are interacting with their repositories, identify trends and patterns in user behavior, and make data-driven decisions to improve the user experience and engagement.
Additionally, the dashboard could be valuable for business analysts, marketing teams, and other stakeholders who are interested in understanding the impact of GitHub on their organization's overall strategy and goals.

### Motivation:
There are multiple motivations behind building this dashboard and not limited too:
1. Improving user experince by analyzing user behavior of GitHub developers.
2. Data-Driven Decision Making by gaining insights to the different segments of the github users and gaining competitive advantage

## Description of the Data
[GitHub Archive](https://www.gharchive.org/) is a project that provides a record of every public event that occurs on GitHub. This includes events such as commits, pull requests, issues, and comments. The archive is a collection of JSON files that are organized by year and month, with each file containing the events that occurred during that time period. The archive is available for free download and can be used for research, analysis, and visualization of GitHub data.
GitHub Archive was created in 2012 by GitHub and the Internet Archive, with the goal of preserving the history of open-source development on GitHub. Since then, it has become a valuable resource for researchers, data scientists, and developers who are interested in analyzing GitHub data for various purposes, such as studying the popularity of programming languages, tracking the growth of open-source projects, or identifying emerging trends in software development.
The data to build this dashboard was originated from [GitHub Archive](https://www.gharchive.org/). Since the size of the data was over 20GB/day, I decided to move forward with the data of March 17th, 2023 data which included over 4 Million events.
The [structure of the datasets](https://github.com/igrigorik/gharchive.org/blob/master/bigquery/schema.js) mentioned above includes separate columns for standard activity fields (as seen in the same response format), a "payload" string field that holds the activity description in JSON encoded format, and an additional "other" string field that encompasses all remaining fields.
After doing a series of data wrangling tasks, the final dataset columns  are the name of an specific event including `Fork`, `Watch`, `PullRequestReview`, `PullRequest` `Create`, `Release`, `Issues`, `Push` and each of the rows are user IDs and the cells contain the count of each event per user.

### Attribution
The data set is public and can be found in [GitHub Archive](https://www.gharchive.org/).


## Research Questions
Main research queries about this could potentialy be:
- What are the most common user behaviors on GitHub, and how do these behaviors vary by user segment (such as developers, project managers, or business analysts)?
- Are there patterns or trends in user behavior depending to their activity type that could be used to identify potential issues or opportunities for improvement in the GitHub user experience?

## Usage Scenario
Here is an example of a usage scenario for a GitHub User Segmentation dashboard:

Let's say that a software development team is working on an open-source project hosted on GitHub, and they are interested in improving the user experience for contributors to the project. They decide to use a GitHub User Segmentation dashboard to gain insights into how users are interacting with the project, and to identify areas for improvement.
After analyzing the data on the dashboard, they discover that a significant portion of their user base consists of developers who are relatively new to GitHub and may not be familiar with the project's workflows or conventions. They also notice that many of these users are not contributing as frequently as more experienced users.
Using this information, the development team decides to implement a series of onboarding resources and tutorials for new GitHub users, with a focus on helping them understand the project's workflows and conventions. They also reach out to the users with the lowest contribution rates to offer personalized assistance and support.
Over time, the team sees an increase in the number of new contributors to the project, as well as an improvement in the engagement and retention rates for new users. The dashboard continues to provide valuable insights that the team uses to make data-driven decisions and improve the overall user experience for their GitHub users.
