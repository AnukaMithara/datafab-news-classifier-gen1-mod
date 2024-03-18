import news_classifier

if __name__ == "__main__":

    news_classifier = news_classifier.NewsClassifier('datafab-classifier.zip')

    # title = " "
    # content = "An Alabama woman seeking in vitro fertilization, Swedish Prime Minister Ulf Kristersson and United Auto Workers president Shawn Fain are among those headed to the US Capitol Thursday evening as President Joe Biden is set to deliver a high-stakes State of the Union address. Biden’s speech could offer his most substantial television audience before voters cast their ballots in the general election, and the remarks are expected to center around the key themes of his reelection campaign, agenda acc"
    language = "en"

    title ="turkey court releases amnesty head taner kilic"
    content="image copyright amnesty international image caption taner kilic accused using encrypted messaging application court istanbul ordered release head human rights group amnesty international turkey detained last june taner kilic charged membership terrorist organisation accusation londonbased group described baseless court ordered release bail ten activists also trial arrests part crackdown following failed coup attempt july 2016 statement amnesty welcomed mr kilics release said would continue pressure charges 11 activists including another amnesty member dropped added unfounded prosecutions attempt silence critical voices within turkey served highlight importance human rights dedicate lives defending image copyright afp image caption activists gathered outside court istanbul mr kilic accused using encrypted messaging application called bylock turkish government said used followers usbased islamic preacher fethullah gulen president recep tayyip erdogan accuses mr gulen behind coup attempt charge cleric denies 40000 people arrested 120000 sacked suspended jobs aftermath failed coup include police military personnel teachers public servants amnesty vocal critic crackdown suspected coup plotters said 2016 credible reports detainees subjected beatings torture including rape"

    print(news_classifier.classify_article(title, content, language))
