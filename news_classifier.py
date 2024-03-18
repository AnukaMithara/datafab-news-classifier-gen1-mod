from classifier.classifier import Classifier
import re


class NewsClassifier:

    """
    NewsClassifier class to classify the news articles
    """

    def __init__(self, file_path):

        """ Constructor to initialize the NewsClassifier class

           Parameters
           ----------
                file_path : str
                    The path to the model file

           Returns
           -------

           Author
           ------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech

           Developers
           ----------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech
        """
        self.classifier = Classifier(file_path)
        self.threshold = 0.5

    @staticmethod
    def clean_text(text):

        """ Clean the text by removing special characters and emojis

           Parameters
           ----------
                text : str
                    The text to be cleaned

           Returns
           -------
                text : str
                    The cleaned text

           Author
           ------
               name: Nipuna Wishwajith
               email: vishwajith@focalid.tech

           Developers
           ----------
               name: Nipuna Wishwajith
               email: vishwajith@focalid.tech
        """
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'[\U00010000-\U0010ffff]', ' ', text)
        return text.strip()

    def classify_article(self, title, content, language="en"):

        """ Classify the news article

           Parameters
           ----------
                title : str
                    The title of the article
                content : str
                    The content of the article
                language : str
                    The language of the article

           Returns
           -------
                categories : list
                    The list of categories of the article

           Author
           ------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech

           Developers
           ----------
               name: Anuka Mithara
               email: karunanayaka@focalid.tech
        """

        model = self.classifier.model
        data = self.clean_text(title + ' ' + content)

        if isinstance(data, str):
            data = [data]
        else:
            raise ValueError('Title and Content must be strings')

        preprocessed_data = self.classifier.transform_preprocessing(data)
        probs = model['model'].predict_proba(preprocessed_data)

        print(probs)

        categories = []
        probs = probs[0]
        for i, prob in enumerate(probs):
            if prob >= self.threshold:
                categories.append(model['model'].classes_[i])
                print(" Index: ", i, " Probability: ", prob, "Category: ", model['model'].classes_[i])

        if not any(categories):
            categories.append('Other')

        print(sum(probs))

        return categories
