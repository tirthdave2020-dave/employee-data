{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/tirthdave2020-dave/employee-data/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sfA-VQMSckmA",
        "outputId": "1b3f515b-96a6-4127-9c7e-11696ecb752f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "         ham       0.98      1.00      0.99       965\n",
            "        spam       0.99      0.89      0.94       150\n",
            "\n",
            "    accuracy                           0.98      1115\n",
            "   macro avg       0.98      0.95      0.96      1115\n",
            "weighted avg       0.98      0.98      0.98      1115\n",
            "\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import classification_report\n",
        "df=pd.read_csv('spam.csv', encoding='latin1')\n",
        "df.head()\n",
        "\n",
        "x=np.array(df['v2'])\n",
        "y=np.array(df['v1'])\n",
        "\n",
        "X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)\n",
        "\n",
        "cv=CountVectorizer()\n",
        "X_train=cv.fit_transform(X_train)\n",
        "X_test=cv.transform(X_test) # Also need to transform X_test\n",
        "\n",
        "model=MultinomialNB()\n",
        "model.fit(X_train,y_train)\n",
        "\n",
        "y_pred=model.predict(X_test)\n",
        "print(classification_report(y_test,y_pred))"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c10c42eb",
        "outputId": "35ddbac7-fb37-430b-8e85-900cd1d82794"
      },
      "source": [
        "import joblib\n",
        "\n",
        "# Save the trained model to a file\n",
        "joblib.dump(model, 'spam_classifier_model.joblib')\n",
        "\n",
        "print(\"Model saved successfully as 'spam_classifier_model.joblib'\")"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model saved successfully as 'spam_classifier_model.joblib'\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOEfrvPhExaTucT1pTO1Scs",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}