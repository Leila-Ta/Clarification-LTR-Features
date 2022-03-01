import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import defaultdict
import ijson
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi

tsv_file = open("./MIMICS-ClickExplore.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

#Prepare features for RankLib Application (https://sourceforge.net/p/lemur/wiki/RankLib/)
out = open("./MIMICS-ClickExplore_analytics.txt", "w")

# outSERP = open("./MIMICS-ClickExplore_analyticsSERP.txt", "w")
# outNonSERP = open("./MIMICS-ClickExplore_analyticsNONSERP.txt", "w")

#Detect all Queries without related search results or/and Video Search results
out_non = open("./MIMICS-ClickExplore_checking_tail.txt", "w")

# out_Duplicateerror = open("./MIMICS-ClickExplore_duplicateerror.txt", "w")

Webpage_Dict= defaultdict(dict)
Video_Dict= defaultdict(dict)
Dict_QueryClarq= defaultdict(list)
options = list()
option1=list()
option2=list()
option3=list()
option4=list()
option5=list()
name_temp=list()
snippet_temp=list()
RS_temp=list()
VN_temp=list()
VD_temp=list()
option_ins = list()
Query=[]
Document_title=[]
snippet=[]
Video_name=[]
Video_description=[]
Related_Search=[]
valu_Query=[]
rsearch_Dict = {}
qid = {}
target_id ={}
qid_number=1
x=0

for row in read_tsv:
    if (x==0):
        x+=1
        continue

    Y = []
    texts = []
    stemmer = WordNetLemmatizer()
    Query = row[0].lower()

    # Remove all the special characters
    Query = re.sub(r'\W', ' ', str(Query))

    # remove all single characters
    Query = re.sub(r'\s+[a-zA-Z]\s+', ' ', Query)

    # Remove single characters from the start
    Query = re.sub(r'\^[a-zA-Z]\s+', ' ', Query)

    # Substituting multiple spaces with single space
    Query = re.sub(r'\s+', ' ', Query, flags=re.I)

    Query = Query.split()
    Query = [stemmer.lemmatize(word) for word in Query]
    Query = ' '.join(Query)

    options.clear()
    #options.append(Query)

    texts.append(row[1].lower())

     # Remove all the special characters
    Y = re.sub(r'\W', ' ', str(texts))

     # remove all single characters
    Y = re.sub(r'\s+[a-zA-Z]\s+', ' ', Y)

     # Remove single characters from the start
    Y = re.sub(r'\^[a-zA-Z]\s+', ' ', Y)

     # Substituting multiple spaces with single space
    Y = re.sub(r'\s+', ' ', Y, flags=re.I)

     # Lemmatization
    Y = Y.split()
    Y = [stemmer.lemmatize(word) for word in Y]
    Y = ' '.join(Y)

    options.append(Y)

    texts = row[2].lower()

     # Remove all the special characters
    texts = re.sub(r'\W', ' ', str(texts))

     # remove all single characters
    texts = re.sub(r'\s+[a-zA-Z]\s+', ' ', texts)

     # Remove single characters from the start
    texts = re.sub(r'\^[a-zA-Z]\s+', ' ', texts)

     # Substituting multiple spaces with single space
    texts = re.sub(r'\s+', ' ', texts, flags=re.I)

    texts = texts.split()

    texts = [stemmer.lemmatize(word) for word in texts]
    texts = ' '.join(texts)

    options.append(texts)

    if (row[3]):
        texts = row[3].lower()

        # Remove all the special characters
        texts = re.sub(r'\W', ' ', str(texts))

        # remove all single characters
        texts = re.sub(r'\s+[a-zA-Z]\s+', ' ', texts)

        # Remove single characters from the start
        texts = re.sub(r'\^[a-zA-Z]\s+', ' ', texts)

        # Substituting multiple spaces with single space
        texts = re.sub(r'\s+', ' ', texts, flags=re.I)

        texts = texts.split()
        texts = [stemmer.lemmatize(word) for word in texts]
        texts = ' '.join(texts)

        options.append(texts)

        if (row[4]):
            texts = row[4].lower()

            # Remove all the special characters
            texts = re.sub(r'\W', ' ', str(texts))

            # remove all single characters
            texts = re.sub(r'\s+[a-zA-Z]\s+', ' ', texts)

            # Remove single characters from the start
            texts = re.sub(r'\^[a-zA-Z]\s+', ' ', texts)

            # Substituting multiple spaces with single space
            texts = re.sub(r'\s+', ' ', texts, flags=re.I)

            texts = texts.split()

            texts = [stemmer.lemmatize(word) for word in texts]
            texts = ' '.join(texts)

            options.append(texts)

            if (row[5]):
                texts = row[5].lower()

                # Remove all the special characters
                texts = re.sub(r'\W', ' ', str(texts))

                # remove all single characters
                texts = re.sub(r'\s+[a-zA-Z]\s+', ' ', texts)

                # Remove single characters from the start
                texts = re.sub(r'\^[a-zA-Z]\s+', ' ', texts)

                # Substituting multiple spaces with single space
                texts = re.sub(r'\s+', ' ', texts, flags=re.I)

                texts = texts.split()
                texts = [stemmer.lemmatize(word) for word in texts]
                texts = ' '.join(texts)

                options.append(texts)

                if (row[6]):
                    texts = row[6].lower()

                    # Remove all the special characters
                    texts = re.sub(r'\W', ' ', str(texts))

                    # remove all single characters
                    texts = re.sub(r'\s+[a-zA-Z]\s+', ' ', texts)

                    # Remove single characters from the start
                    texts = re.sub(r'\^[a-zA-Z]\s+', ' ', texts)

                    # Substituting multiple spaces with single space
                    texts = re.sub(r'\s+', ' ', texts, flags=re.I)

                    texts = texts.split()
                    texts = [stemmer.lemmatize(word) for word in texts]
                    texts = ' '.join(texts)

                    options.append(texts)

    if (row[0] not in Dict_QueryClarq.keys()):
        Dict_QueryClarq[row[0]]=[options.copy()]
        qid[row[0]]=qid_number
        qid_number+=1
    else:
        option_ins.clear()
        fag=0

        for z in [Dict_QueryClarq[row[0]]]:
            if (type(z[0]) == list):
                for zz in z:
                    if (zz == options):
                        fag=1
                        continue
                    else:
                        option_ins = (option_ins + [zz]).copy()
            else:
                if (z == options):
                    fag=1
                    break
                else:
                    option_ins = (option_ins + [z]).copy()

        if (fag==1):
            fag=0
            continue
        else:
            option_ins=(option_ins+[options.copy()]).copy()
            Dict_QueryClarq[row[0]]=option_ins.copy()

    target_id[tuple(options)] = x
    x += 1

tsv_file.close()

del texts
del options

fl=0

#SERP
with open("/Users/leila/PhD/Clarifying Question/MIMICS-BingAPI.result") as f:
    for line in f:
        for prefix, type_of_object, value in ijson.parse(line):
            if prefix == 'queryContext.originalQuery':
                Querycontext=value

                if (Querycontext not in Dict_QueryClarq.keys()):
                    fl=1
                    break
                else:
                    if len(Dict_QueryClarq[Querycontext]) == 1:
                        # out_Duplicateerror.write(Querycontext)
                        # out_Duplicateerror.write('\n')
                        fl = 1
                        break

                    print("Main Query:")
                    print(Querycontext)

            if prefix == 'webPages.value.item.name':
                Document_title=value.lower()

                print("Document_title")
                print(Document_title)

                  # Remove all the special characters
                Document_title = re.sub(r'\W', ' ', str(Document_title))

                  # remove all single characters
                Document_title = re.sub(r'\s+[a-zA-Z]\s+', ' ', Document_title)

                  # Remove single characters from the start
                Document_title = re.sub(r'\^[a-zA-Z]\s+', ' ', Document_title)

                  # Substituting multiple spaces with single space
                Document_title = re.sub(r'\s+', ' ', Document_title, flags=re.I)

                Document_title = Document_title.split()
                Document_title = [stemmer.lemmatize(word) for word in Document_title]
                Document_title = ' '.join(Document_title)

            if prefix == 'webPages.value.item.snippet':
                snippet = value.lower()

                print("Snippet")
                print(snippet)

                  # Remove all the special characters
                snippet = re.sub(r'\W', ' ', str(snippet))

                 # remove all single characters
                snippet = re.sub(r'\s+[a-zA-Z]\s+', ' ', snippet)

                  # Remove single characters from the start
                snippet = re.sub(r'\^[a-zA-Z]\s+', ' ', snippet)

                  # Substituting multiple spaces with single space
                snippet = re.sub(r'\s+', ' ', snippet, flags=re.I)



                snippet = snippet.split()
                snippet = [stemmer.lemmatize(word) for word in snippet]
                snippet = ' '.join(snippet)

                Webpage_Dict[Querycontext][Document_title] = snippet

            if prefix == 'relatedSearches.value.item.text':
                Related_Search=value.lower()

                 # Substituting multiple spaces with single space
                Related_Search = re.sub(r'\s+', ' ', Related_Search, flags=re.I)
                Related_Search = Related_Search.split()
                Related_Search = [stemmer.lemmatize(word) for word in Related_Search]
                Related_Search = ' '.join(Related_Search)
                option_ins.clear()

                if (Querycontext not in rsearch_Dict.keys()):
                    rsearch_Dict[Querycontext]= [Related_Search]
                else:
                    for z in rsearch_Dict[Querycontext]:
                        option_ins = (option_ins + [z]).copy()
                        option_ins = (option_ins + [Related_Search]).copy()
                        rsearch_Dict[Querycontext] = option_ins.copy()

            if prefix == 'videos.value.item.name':
                Video_name = value.lower()

                 # Remove all the special characters
                Video_name = re.sub(r'\W', ' ', str(Video_name))

                 # remove all single characters
                Video_name = re.sub(r'\s+[a-zA-Z]\s+', ' ', Video_name)

                 # Remove single characters from the start
                Video_name = re.sub(r'\^[a-zA-Z]\s+', ' ', Video_name)

                 # Substituting multiple spaces with single space
                Video_name = re.sub(r'\s+', ' ', Video_name, flags=re.I)

                Video_name = Video_name.split()
                Video_name = [stemmer.lemmatize(word) for word in Video_name]
                Video_name = ' '.join(Video_name)

            if prefix == 'videos.value.item.description':
                Video_description = value.lower()

                 # Remove all the special characters
                Video_description = re.sub(r'\W', ' ', str(Video_description))

                 # remove all single characters
                Video_description = re.sub(r'\s+[a-zA-Z]\s+', ' ', Video_description)

                 # Remove single characters from the start
                Video_description = re.sub(r'\^[a-zA-Z]\s+', ' ', Video_description)

                 # Substituting multiple spaces with single space
                Video_description = re.sub(r'\s+', ' ', Video_description, flags=re.I)

                Video_description = Video_description.split()
                Video_description = [stemmer.lemmatize(word) for word in Video_description]
                Video_description = ' '.join(Video_description)
                Video_Dict[Querycontext][Video_name] = Video_description

        if (fl==1):
            fl=0
            continue
        else:
            vectorizer = TfidfVectorizer()
            name_temp.clear()
            snippet_temp.clear()
            RS_temp.clear()
            VN_temp.clear()
            VD_temp.clear()

            if Querycontext in Webpage_Dict.keys():
                for D_T, Snip in Webpage_Dict[Querycontext].items():
                    name_temp.append(D_T)
                    snippet_temp.append(Snip)

                tokenized_name = [doc.split(' ') for doc in name_temp]
                tokenized_snippet = [doc.split(' ') for doc in snippet_temp]
            else:
                continue

            if Querycontext in rsearch_Dict.keys():
                for RS in rsearch_Dict[Querycontext]:
                    RS_temp.append(RS)

                tokenized_RS = [doc.split(' ') for doc in RS_temp]
            else:
                out_non.write("RelatedSearch\t" + Querycontext)
                out_non.write('\n')

            if Querycontext in Video_Dict.keys():
                for VN, VD in Video_Dict[Querycontext].items():
                    VN_temp.append(VN)
                    VD_temp.append(VD)

                tokenized_VN = [doc.split(' ') for doc in VN_temp]
                tokenized_VD = [doc.split(' ') for doc in VD_temp]
            else:
                out_non.write("Video\t" + Querycontext)
                out_non.write('\n')

            Query = Querycontext.lower()

            # Remove all the special characters
            Query = re.sub(r'\W', ' ', str(Query))

            # remove all single characters
            Query = re.sub(r'\s+[a-zA-Z]\s+', ' ', Query)

            # Remove single characters from the start
            Query = re.sub(r'\^[a-zA-Z]\s+', ' ', Query)

            # Substituting multiple spaces with single space
            Query = re.sub(r'\s+', ' ', Query, flags=re.I)

            Query = Query.split()
            Query = [stemmer.lemmatize(word) for word in Query]
            Query = ' '.join(Query)

            tokenized_QU = Query.split(' ')

            querytempnumber = 1

            for j in tokenized_QU:
                if querytempnumber < len(j):
                    querytempnumber = len(j)

                if (querytempnumber > 1):
                    break

            for valu in Dict_QueryClarq[Querycontext]:
                print("valu")
                print(valu)

                option1.clear()
                option2.clear()
                option3.clear()
                option4.clear()
                option5.clear()

                option1 = valu.copy()
                option2 = valu.copy()

                b = len(valu)-1
                print("here")
                print(b)


                while(b>1):
                    option1.pop(b)
                    b-=1

                option2.pop(1)
                b = len(valu) - 2

                while (b > 1):
                    option2.pop(b)
                    b -= 1

                if(len(valu)>3):
                    option3 = valu.copy()
                    option3.pop(1)
                    option3.pop(1)
                    b = len(valu) - 3

                    while (b > 1):
                        option3.pop(b)
                        b -= 1

                    if (len(valu) > 4):
                        option4 = valu.copy()
                        option4.pop(1)
                        option4.pop(1)
                        option4.pop(1)
                        b = len(valu) - 4

                        while (b > 1):
                            option4.pop(b)
                            b -= 1

                        if (len(valu) > 5):
                            option5 = valu.copy()
                            option5.pop(1)
                            option5.pop(1)
                            option5.pop(1)
                            option5.pop(1)

                print(option1)
                print(option2)
                print(option3)
                print(option4)
                print(option5)
                valu_Query = valu.copy()
                print(valu_Query)

                strin = ""
                j = 0

                for i in valu:
                    strin = strin + i
                    j += 1

                    if j < len(valu):
                        strin = strin + " "

                tokenized_valu = strin.split(" ")

                stri = ""
                cc=0

                for k in valu_Query:
                    stri = stri + k
                    cc += 1

                    if cc < len(valu_Query):
                        stri = stri + " "

                tokenized_valu_Query = stri.split(" ")

                cosinetfidfDT_num = 0
                cosinetfidfSN_num = 0
                fu_DT=0
                fu_SN=0
                DT_scores= [0]
                SN_scores= [0]

                print("nametemp")
                print(name_temp)

                DT_VECTFIDF = vectorizer.fit_transform(name_temp)
                query_vec = vectorizer.transform(valu)
                print(query_vec)
                results_DT = cosine_similarity(DT_VECTFIDF, query_vec)
                print(results_DT)
                bm25_DT = BM25Okapi(tokenized_name)
                DT_scores = bm25_DT.get_scores(tokenized_valu)
                SN_VECTFIDF = vectorizer.fit_transform(snippet_temp)
                query_vec = vectorizer.transform(valu)
                results_SN = cosine_similarity(SN_VECTFIDF, query_vec)
                bm25_SN = BM25Okapi(tokenized_snippet)
                SN_scores = bm25_SN.get_scores(tokenized_valu)
                cosinetfidfDT_num = (np.sum(results_DT)) / len(results_DT)
                cosinetfidfSN_num = (np.sum(results_SN)) / len(results_SN)
                fu_DT = (fuzz.token_sort_ratio(valu, name_temp))
                fu_SN = (fuzz.token_sort_ratio(valu, snippet_temp))

                cosinetfidfRS_num = 0
                fu_RS=0
                RS_scores=[0]

                if Querycontext in rsearch_Dict.keys():
                    RS_VECTFIDF = vectorizer.fit_transform(RS_temp)
                    query_vec = vectorizer.transform(valu)
                    results_RS = cosine_similarity(RS_VECTFIDF, query_vec)
                    cosinetfidfRS_num = (np.sum(results_RS)) / len(results_RS)
                    bm25_RS = BM25Okapi(tokenized_RS)
                    RS_scores = bm25_RS.get_scores(tokenized_valu)
                    fu_RS = (fuzz.token_sort_ratio(valu, RS_temp))

                cosinetfidfVN_num = 0
                cosinetfidfVD_num=0
                fu_VN=0
                fu_VD=0
                VN_scores = [0]
                VD_scores = [0]

                if Querycontext in Video_Dict.keys():
                    VN_VECTFIDF = vectorizer.fit_transform(VN_temp)
                    query_vec = vectorizer.transform(valu)
                    results_VN = cosine_similarity(VN_VECTFIDF, query_vec)
                    cosinetfidfVN_num = (np.sum(results_VN)) / len(results_VN)
                    bm25_VN = BM25Okapi(tokenized_VN)
                    VN_scores = bm25_VN.get_scores(tokenized_valu)
                    VD_VECTFIDF = vectorizer.fit_transform(VD_temp)
                    query_vec = vectorizer.transform(valu)
                    results_VD = cosine_similarity(VD_VECTFIDF, query_vec)
                    cosinetfidfVD_num = (np.sum(results_VD)) / len(results_VD)
                    bm25_VD = BM25Okapi(tokenized_VD)
                    VD_scores = bm25_VD.get_scores(tokenized_valu)
                    fu_VN = (fuzz.token_sort_ratio(valu, VN_temp))
                    fu_VD = (fuzz.token_sort_ratio(valu, VD_temp))

                cosinetfidfQU_num = 0
                fu_QU = 0
                QU_scores = [0]

                if (querytempnumber>1):
                    QU_VECTFIDF = vectorizer.fit_transform([Query])
                    query_vec = vectorizer.transform(valu_Query)
                    results_QU = cosine_similarity(QU_VECTFIDF, query_vec)
                    cosinetfidfQU_num = (np.sum(results_QU)) / len(results_QU)
                    bm25_QU = BM25Okapi([tokenized_valu_Query])
                    QU_scores = bm25_QU.get_scores(tokenized_QU)

                    fu_QU = (fuzz.token_sort_ratio(valu_Query, Query))

                #Option1
                option1_Query = option1.copy()

                strin = ""
                j = 0

                for i in option1:
                    strin = strin + i
                    j += 1

                    if j < len(option1):
                        strin = strin + " "

                tokenized_option1 = strin.split(" ")

                stri = ""
                cc=0

                for k in option1_Query:
                    stri = stri + k
                    cc += 1

                    if cc < len(option1_Query):
                        stri = stri + " "

                tokenized_option1_Query = stri.split(" ")

                cosinetfidfDT1_num = 0
                cosinetfidfSN1_num = 0
                fu_DT1=0
                fu_SN1 = 0
                DT_scores1= [0]
                SN_scores1= [0]

                DT_VECTFIDF = vectorizer.fit_transform(name_temp)
                query_vec = vectorizer.transform(option1)
                results_DT = cosine_similarity(DT_VECTFIDF, query_vec)
                bm25_DT = BM25Okapi(tokenized_name)
                DT_scores1 = bm25_DT.get_scores(tokenized_option1)
                SN_VECTFIDF = vectorizer.fit_transform(snippet_temp)
                query_vec = vectorizer.transform(option1)
                results_SN = cosine_similarity(SN_VECTFIDF, query_vec)
                bm25_SN = BM25Okapi(tokenized_snippet)
                SN_scores1 = bm25_SN.get_scores(tokenized_option1)
                cosinetfidfDT1_num = (np.sum(results_DT)) / len(results_DT)
                cosinetfidfSN1_num = (np.sum(results_SN)) / len(results_SN)
                fu_DT1 = (fuzz.token_sort_ratio(option1, name_temp))
                fu_SN1 = (fuzz.token_sort_ratio(option1, snippet_temp))

                cosinetfidfRS1_num = 0
                fu_RS1 =0
                RS_scores1=[0]

                if Querycontext in rsearch_Dict.keys():
                    RS_VECTFIDF = vectorizer.fit_transform(RS_temp)
                    query_vec = vectorizer.transform(option1)
                    results_RS = cosine_similarity(RS_VECTFIDF, query_vec)
                    cosinetfidfRS1_num = (np.sum(results_RS)) / len(results_RS)
                    bm25_RS = BM25Okapi(tokenized_RS)
                    RS_scores1 = bm25_RS.get_scores(tokenized_option1)
                    fu_RS1 = (fuzz.token_sort_ratio(option1, RS_temp))

                cosinetfidfVN1_num = 0
                cosinetfidfVD1_num=0
                fu_VN1=0
                fu_VD1=0
                VN_scores1 = [0]
                VD_scores1 = [0]

                if Querycontext in Video_Dict.keys():
                    VN_VECTFIDF = vectorizer.fit_transform(VN_temp)
                    query_vec = vectorizer.transform(option1)
                    results_VN = cosine_similarity(VN_VECTFIDF, query_vec)
                    cosinetfidfVN1_num = (np.sum(results_VN)) / len(results_VN)
                    bm25_VN = BM25Okapi(tokenized_VN)
                    VN_scores1 = bm25_VN.get_scores(tokenized_option1)
                    VD_VECTFIDF = vectorizer.fit_transform(VD_temp)
                    query_vec = vectorizer.transform(option1)
                    results_VD = cosine_similarity(VD_VECTFIDF, query_vec)
                    cosinetfidfVD1_num = (np.sum(results_VD)) / len(results_VD)
                    bm25_VD = BM25Okapi(tokenized_VD)
                    VD_scores1 = bm25_VD.get_scores(tokenized_option1)
                    fu_VN1 = (fuzz.token_sort_ratio(option1, VN_temp))
                    fu_VD1 = (fuzz.token_sort_ratio(option1, VD_temp))

                cosinetfidfQU1_num = 0
                fu_QU1 = 0
                QU_scores1 = [0]

                if (querytempnumber>1):
                    QU_VECTFIDF = vectorizer.fit_transform([Query])
                    query_vec = vectorizer.transform(option1_Query)
                    results_QU = cosine_similarity(QU_VECTFIDF, query_vec)
                    cosinetfidfQU1_num = (np.sum(results_QU)) / len(results_QU)
                    bm25_QU = BM25Okapi([tokenized_option1_Query])
                    QU_scores1 = bm25_QU.get_scores(tokenized_QU)

                    fu_QU1 = (fuzz.token_sort_ratio(option1_Query, Query))

                #Option2
                option2_Query = option2.copy()

                strin = ""
                j = 0

                for i in option2:
                    strin = strin + i
                    j += 1

                    if j < len(option2):
                        strin = strin + " "

                tokenized_option2 = strin.split(" ")

                stri = ""
                cc=0

                for k in option2_Query:
                    stri = stri + k
                    cc += 1

                    if cc < len(option2_Query):
                        stri = stri + " "

                tokenized_option2_Query = stri.split(" ")

                cosinetfidfDT2_num = 0
                cosinetfidfSN2_num = 0
                fu_DT2 = 0
                fu_SN2 = 0
                DT_scores2= [0]
                SN_scores2= [0]

                DT_VECTFIDF = vectorizer.fit_transform(name_temp)
                query_vec = vectorizer.transform(option2)
                results_DT = cosine_similarity(DT_VECTFIDF, query_vec)
                bm25_DT = BM25Okapi(tokenized_name)
                DT_scores2 = bm25_DT.get_scores(tokenized_option2)
                SN_VECTFIDF = vectorizer.fit_transform(snippet_temp)
                query_vec = vectorizer.transform(option2)
                results_SN = cosine_similarity(SN_VECTFIDF, query_vec)
                bm25_SN = BM25Okapi(tokenized_snippet)
                SN_scores2 = bm25_SN.get_scores(tokenized_option2)
                cosinetfidfDT2_num = (np.sum(results_DT)) / len(results_DT)
                cosinetfidfSN2_num = (np.sum(results_SN)) / len(results_SN)
                fu_DT2 = (fuzz.token_sort_ratio(option2, name_temp))
                fu_SN2 = (fuzz.token_sort_ratio(option2, snippet_temp))

                cosinetfidfRS2_num = 0
                fu_RS2 =0
                RS_scores2=[0]

                if Querycontext in rsearch_Dict.keys():
                    RS_VECTFIDF = vectorizer.fit_transform(RS_temp)
                    query_vec = vectorizer.transform(option2)
                    results_RS = cosine_similarity(RS_VECTFIDF, query_vec)
                    cosinetfidfRS2_num = (np.sum(results_RS)) / len(results_RS)
                    bm25_RS = BM25Okapi(tokenized_RS)
                    RS_scores2 = bm25_RS.get_scores(tokenized_option2)
                    fu_RS2 = (fuzz.token_sort_ratio(option2, RS_temp))

                cosinetfidfVN2_num = 0
                cosinetfidfVD2_num=0
                fu_VN2=0
                fu_VD2=0
                VN_scores2 = [0]
                VD_scores2 = [0]

                if Querycontext in Video_Dict.keys():
                    VN_VECTFIDF = vectorizer.fit_transform(VN_temp)
                    query_vec = vectorizer.transform(option2)
                    results_VN = cosine_similarity(VN_VECTFIDF, query_vec)
                    cosinetfidfVN2_num = (np.sum(results_VN)) / len(results_VN)
                    bm25_VN = BM25Okapi(tokenized_VN)
                    VN_scores2 = bm25_VN.get_scores(tokenized_option2)
                    VD_VECTFIDF = vectorizer.fit_transform(VD_temp)
                    query_vec = vectorizer.transform(option2)
                    results_VD = cosine_similarity(VD_VECTFIDF, query_vec)
                    cosinetfidfVD2_num = (np.sum(results_VD)) / len(results_VD)
                    bm25_VD = BM25Okapi(tokenized_VD)
                    VD_scores2 = bm25_VD.get_scores(tokenized_option2)
                    fu_VN2 = (fuzz.token_sort_ratio(option2, VN_temp))
                    fu_VD2 = (fuzz.token_sort_ratio(option2, VD_temp))

                cosinetfidfQU2_num = 0
                fu_QU2 = 0
                QU_scores2 = [0]

                if (querytempnumber>1):
                    QU_VECTFIDF = vectorizer.fit_transform([Query])
                    query_vec = vectorizer.transform(option2_Query)
                    results_QU = cosine_similarity(QU_VECTFIDF, query_vec)
                    cosinetfidfQU2_num = (np.sum(results_QU)) / len(results_QU)
                    bm25_QU = BM25Okapi([tokenized_option2_Query])
                    QU_scores2 = bm25_QU.get_scores(tokenized_QU)

                    fu_QU2 = (fuzz.token_sort_ratio(option2_Query, Query))

                cosinetfidfDT3_num = 0
                cosinetfidfSN3_num = 0
                fu_DT3 = 0
                fu_SN3 = 0
                DT_scores3 = [0]
                SN_scores3 = [0]
                cosinetfidfRS3_num = 0
                fu_RS3 = 0
                RS_scores3 = [0]
                cosinetfidfVN3_num = 0
                cosinetfidfVD3_num = 0
                fu_VN3 = 0
                fu_VD3 = 0
                VN_scores3 = [0]
                VD_scores3 = [0]
                cosinetfidfQU3_num = 0
                fu_QU3 = 0
                QU_scores3 = [0]

                cosinetfidfDT4_num = 0
                cosinetfidfSN4_num = 0
                fu_DT4 = 0
                fu_SN4 = 0
                DT_scores4 = [0]
                SN_scores4 = [0]
                cosinetfidfRS4_num = 0
                fu_RS4 = 0
                RS_scores4 = [0]
                cosinetfidfVN4_num = 0
                cosinetfidfVD4_num = 0
                fu_VN4 = 0
                fu_VD4 = 0
                VN_scores4 = [0]
                VD_scores4 = [0]
                cosinetfidfQU4_num = 0
                fu_QU4 = 0
                QU_scores4 = [0]
                cosinetfidfDT5_num = 0
                cosinetfidfSN5_num = 0
                fu_DT5 = 0
                fu_SN5 = 0
                DT_scores5 = [0]
                SN_scores5 = [0]
                cosinetfidfRS5_num = 0
                fu_RS5 = 0
                RS_scores5 = [0]
                cosinetfidfVN5_num = 0
                cosinetfidfVD5_num = 0
                fu_VN5 = 0
                fu_VD5 = 0
                VN_scores5 = [0]
                VD_scores5 = [0]
                cosinetfidfQU5_num = 0
                fu_QU5 = 0
                QU_scores5 = [0]


                #Option3
                if (option3):
                    option3_Query = option3.copy()

                    strin = ""
                    j = 0

                    for i in option3:
                        strin = strin + i
                        j += 1

                        if j < len(option3):
                            strin = strin + " "

                    tokenized_option3 = strin.split(" ")

                    stri = ""
                    cc = 0

                    for k in option3_Query:
                        stri = stri + k
                        cc += 1

                        if cc < len(option3_Query):
                            stri = stri + " "

                    tokenized_option3_Query = stri.split(" ")

                    if Querycontext in Webpage_Dict.keys():
                        DT_VECTFIDF = vectorizer.fit_transform(name_temp)
                        query_vec = vectorizer.transform(option3)
                        results_DT = cosine_similarity(DT_VECTFIDF, query_vec)
                        bm25_DT = BM25Okapi(tokenized_name)
                        DT_scores3 = bm25_DT.get_scores(tokenized_option3)
                        SN_VECTFIDF = vectorizer.fit_transform(snippet_temp)
                        query_vec = vectorizer.transform(option3)
                        results_SN = cosine_similarity(SN_VECTFIDF, query_vec)
                        bm25_SN = BM25Okapi(tokenized_snippet)
                        SN_scores3 = bm25_SN.get_scores(tokenized_option3)
                        cosinetfidfDT3_num = (np.sum(results_DT)) / len(results_DT)
                        cosinetfidfSN3_num = (np.sum(results_SN)) / len(results_SN)
                        fu_DT3 = (fuzz.token_sort_ratio(option3, name_temp))
                        fu_SN3 = (fuzz.token_sort_ratio(option3, snippet_temp))

                    if Querycontext in rsearch_Dict.keys():
                        RS_VECTFIDF = vectorizer.fit_transform(RS_temp)
                        query_vec = vectorizer.transform(option3)
                        results_RS = cosine_similarity(RS_VECTFIDF, query_vec)
                        cosinetfidfRS3_num = (np.sum(results_RS)) / len(results_RS)
                        bm25_RS = BM25Okapi(tokenized_RS)
                        RS_scores3 = bm25_RS.get_scores(tokenized_option3)
                        fu_RS3 = (fuzz.token_sort_ratio(option3, RS_temp))

                    if Querycontext in Video_Dict.keys():
                        VN_VECTFIDF = vectorizer.fit_transform(VN_temp)
                        query_vec = vectorizer.transform(option3)
                        results_VN = cosine_similarity(VN_VECTFIDF, query_vec)
                        cosinetfidfVN3_num = (np.sum(results_VN)) / len(results_VN)
                        bm25_VN = BM25Okapi(tokenized_VN)
                        VN_scores3 = bm25_VN.get_scores(tokenized_option3)
                        VD_VECTFIDF = vectorizer.fit_transform(VD_temp)
                        query_vec = vectorizer.transform(option3)
                        results_VD = cosine_similarity(VD_VECTFIDF, query_vec)
                        cosinetfidfVD3_num = (np.sum(results_VD)) / len(results_VD)
                        bm25_VD = BM25Okapi(tokenized_VD)
                        VD_scores3 = bm25_VD.get_scores(tokenized_option3)
                        fu_VN3 = (fuzz.token_sort_ratio(option3, VN_temp))
                        fu_VD3 = (fuzz.token_sort_ratio(option3, VD_temp))

                    if (querytempnumber > 1):
                        QU_VECTFIDF = vectorizer.fit_transform([Query])
                        query_vec = vectorizer.transform(option3_Query)
                        results_QU = cosine_similarity(QU_VECTFIDF, query_vec)
                        cosinetfidfQU3_num = (np.sum(results_QU)) / len(results_QU)
                        bm25_QU = BM25Okapi([tokenized_option3_Query])
                        QU_scores3 = bm25_QU.get_scores(tokenized_QU)

                        fu_QU3 = (fuzz.token_sort_ratio(option3_Query, Query))

                    # Option4
                    if (option4):
                        option4_Query = option4.copy()

                        strin = ""
                        j = 0

                        for i in option4:
                            strin = strin + i
                            j += 1

                            if j < len(option4):
                                strin = strin + " "

                        tokenized_option4 = strin.split(" ")

                        stri = ""
                        cc = 0

                        for k in option4_Query:
                            stri = stri + k
                            cc += 1

                            if cc < len(option4_Query):
                                stri = stri + " "

                        tokenized_option4_Query = stri.split(" ")

                        if Querycontext in Webpage_Dict.keys():
                            DT_VECTFIDF = vectorizer.fit_transform(name_temp)
                            query_vec = vectorizer.transform(option4)
                            results_DT = cosine_similarity(DT_VECTFIDF, query_vec)
                            bm25_DT = BM25Okapi(tokenized_name)
                            DT_scores4 = bm25_DT.get_scores(tokenized_option4)
                            SN_VECTFIDF = vectorizer.fit_transform(snippet_temp)
                            query_vec = vectorizer.transform(option4)
                            results_SN = cosine_similarity(SN_VECTFIDF, query_vec)
                            bm25_SN = BM25Okapi(tokenized_snippet)
                            SN_scores4 = bm25_SN.get_scores(tokenized_option4)
                            cosinetfidfDT4_num = (np.sum(results_DT)) / len(results_DT)
                            cosinetfidfSN4_num = (np.sum(results_SN)) / len(results_SN)
                            fu_DT4 = (fuzz.token_sort_ratio(option4, name_temp))
                            fu_SN4 = (fuzz.token_sort_ratio(option4, snippet_temp))

                        if Querycontext in rsearch_Dict.keys():
                            RS_VECTFIDF = vectorizer.fit_transform(RS_temp)
                            query_vec = vectorizer.transform(option4)
                            results_RS = cosine_similarity(RS_VECTFIDF, query_vec)
                            cosinetfidfRS4_num = (np.sum(results_RS)) / len(results_RS)
                            bm25_RS = BM25Okapi(tokenized_RS)
                            RS_scores4 = bm25_RS.get_scores(tokenized_option4)
                            fu_RS4 = (fuzz.token_sort_ratio(option4, RS_temp))

                        if Querycontext in Video_Dict.keys():
                            VN_VECTFIDF = vectorizer.fit_transform(VN_temp)
                            query_vec = vectorizer.transform(option4)
                            results_VN = cosine_similarity(VN_VECTFIDF, query_vec)
                            cosinetfidfVN4_num = (np.sum(results_VN)) / len(results_VN)
                            bm25_VN = BM25Okapi(tokenized_VN)
                            VN_scores4 = bm25_VN.get_scores(tokenized_option4)
                            VD_VECTFIDF = vectorizer.fit_transform(VD_temp)
                            query_vec = vectorizer.transform(option4)
                            results_VD = cosine_similarity(VD_VECTFIDF, query_vec)
                            cosinetfidfVD4_num = (np.sum(results_VD)) / len(results_VD)
                            bm25_VD = BM25Okapi(tokenized_VD)
                            VD_scores4 = bm25_VD.get_scores(tokenized_option4)
                            fu_VN4 = (fuzz.token_sort_ratio(option4, VN_temp))
                            fu_VD4 = (fuzz.token_sort_ratio(option4, VD_temp))

                        if (querytempnumber > 1):
                            QU_VECTFIDF = vectorizer.fit_transform([Query])
                            query_vec = vectorizer.transform(option4_Query)
                            results_QU = cosine_similarity(QU_VECTFIDF, query_vec)
                            cosinetfidfQU4_num = (np.sum(results_QU)) / len(results_QU)
                            bm25_QU = BM25Okapi([tokenized_option4_Query])
                            QU_scores4 = bm25_QU.get_scores(tokenized_QU)

                            fu_QU4 = (fuzz.token_sort_ratio(option4_Query, Query))

                        # Option5
                        if(option5):
                            option5_Query = option5.copy()

                            strin = ""
                            j = 0

                            for i in option5:
                                strin = strin + i
                                j += 1

                                if j < len(option5):
                                    strin = strin + " "

                            tokenized_option5 = strin.split(" ")

                            stri = ""
                            cc = 0

                            for k in option5_Query:
                                stri = stri + k
                                cc += 1

                                if cc < len(option5_Query):
                                    stri = stri + " "

                            tokenized_option5_Query = stri.split(" ")

                            if Querycontext in Webpage_Dict.keys():
                                DT_VECTFIDF = vectorizer.fit_transform(name_temp)
                                query_vec = vectorizer.transform(option5)
                                results_DT = cosine_similarity(DT_VECTFIDF, query_vec)
                                bm25_DT = BM25Okapi(tokenized_name)
                                DT_scores5 = bm25_DT.get_scores(tokenized_option5)
                                SN_VECTFIDF = vectorizer.fit_transform(snippet_temp)
                                query_vec = vectorizer.transform(option5)
                                results_SN = cosine_similarity(SN_VECTFIDF, query_vec)
                                bm25_SN = BM25Okapi(tokenized_snippet)
                                SN_scores5 = bm25_SN.get_scores(tokenized_option5)
                                cosinetfidfDT5_num = (np.sum(results_DT)) / len(results_DT)
                                cosinetfidfSN5_num = (np.sum(results_SN)) / len(results_SN)
                                fu_DT5 = (fuzz.token_sort_ratio(option5, name_temp))
                                fu_SN5 = (fuzz.token_sort_ratio(option5, snippet_temp))

                            if Querycontext in rsearch_Dict.keys():
                                RS_VECTFIDF = vectorizer.fit_transform(RS_temp)
                                query_vec = vectorizer.transform(option5)
                                results_RS = cosine_similarity(RS_VECTFIDF, query_vec)
                                cosinetfidfRS5_num = (np.sum(results_RS)) / len(results_RS)
                                bm25_RS = BM25Okapi(tokenized_RS)
                                RS_scores5 = bm25_RS.get_scores(tokenized_option5)
                                fu_RS5 = (fuzz.token_sort_ratio(option5, RS_temp))

                            if Querycontext in Video_Dict.keys():
                                VN_VECTFIDF = vectorizer.fit_transform(VN_temp)
                                query_vec = vectorizer.transform(option5)
                                results_VN = cosine_similarity(VN_VECTFIDF, query_vec)
                                cosinetfidfVN5_num = (np.sum(results_VN)) / len(results_VN)
                                bm25_VN = BM25Okapi(tokenized_VN)
                                VN_scores5 = bm25_VN.get_scores(tokenized_option5)
                                VD_VECTFIDF = vectorizer.fit_transform(VD_temp)
                                query_vec = vectorizer.transform(option5)
                                results_VD = cosine_similarity(VD_VECTFIDF, query_vec)
                                cosinetfidfVD5_num = (np.sum(results_VD)) / len(results_VD)
                                bm25_VD = BM25Okapi(tokenized_VD)
                                VD_scores5 = bm25_VD.get_scores(tokenized_option5)
                                fu_VN5 = (fuzz.token_sort_ratio(option5, VN_temp))
                                fu_VD5 = (fuzz.token_sort_ratio(option5, VD_temp))

                            if (querytempnumber > 1):
                                QU_VECTFIDF = vectorizer.fit_transform([Query])
                                query_vec = vectorizer.transform(option5_Query)
                                results_QU = cosine_similarity(QU_VECTFIDF, query_vec)
                                cosinetfidfQU5_num = (np.sum(results_QU)) / len(results_QU)
                                bm25_QU = BM25Okapi([tokenized_option5_Query])
                                QU_scores5 = bm25_QU.get_scores(tokenized_QU)

                                fu_QU5 = (fuzz.token_sort_ratio(option5_Query, Query))

                out.write(str(target_id[tuple(valu)]) + " qid:" + str(qid[Querycontext]))
                # outSERP.write(str(target_id[tuple(valu)]) + " qid:" + str(qid[Querycontext]))
                # outNonSERP.write(str(target_id[tuple(valu)]) + " qid:" + str(qid[Querycontext]))

                #DocumentTitleFeature(CosineSimilarityTFIDF)
                out.write(" 1:" + str(cosinetfidfDT_num))
                # outSERP.write(" 1:" + str(cosinetfidfDT_num))

                # DocumentSnippetFeature(CosineSimilarityTFIDF)
                out.write(" 2:" + str(cosinetfidfSN_num))
                # outSERP.write(" 2:" + str(cosinetfidfSN_num))

                # RelatedSearchFeature(CosineSimilarityTFIDF)
                out.write(" 3:" + str(cosinetfidfRS_num))
                # outSERP.write(" 3:" + str(cosinetfidfRS_num))

                # VideoTitleFeature(CosineSimilarityTFIDF)
                out.write(" 4:" + str(cosinetfidfVN_num))
                # outSERP.write(" 4:" + str(cosinetfidfVN_num))

                # VideDescriptionFeature(CosineSimilarityTFIDF)
                out.write(" 5:" + str(cosinetfidfVD_num))
                # outSERP.write(" 5:" + str(cosinetfidfVD_num))

                # DocumentTitleFeature(NumberofTermMatching)
                out.write(" 6:" + str(fu_DT))
                # outSERP.write(" 6:" + str(fu_DT))

                # DocumentSnippetFeature(NumberofTermMatching)
                out.write(" 7:" + str(fu_SN))
                # outSERP.write(" 7:" + str(fu_SN))

                # RelatedSearchFeature(NumberofTermMatching)
                out.write(" 8:" + str(fu_RS))
                # outSERP.write(" 8:" + str(fu_RS))

                # VideoTitleFeature(NumberofTermMatching)
                out.write(" 9:" + str(fu_VN))
                # outSERP.write(" 9:" + str(fu_VN))

                # VideoDescriptionFeature(NumberofTermMatching)
                out.write(" 10:" + str(fu_VD))
                # outSERP.write(" 10:" + str(fu_VD))

                # DocumentTitleFeature(BM25)
                out.write(" 11:" + str(np.sum(DT_scores) / len(DT_scores)))
                # outSERP.write(" 11:" + str(np.sum(DT_scores) / len(DT_scores)))

                # DocumentSnippetFeature(BM25)
                out.write(" 12:" + str(np.sum(SN_scores) / len(SN_scores)))
                # outSERP.write(" 12:" + str(np.sum(SN_scores) / len(SN_scores)))

                # RelatedSearchFeature(BM25)
                out.write(" 13:" + str(np.sum(RS_scores) / len(RS_scores)))
                # outSERP.write(" 13:" + str(np.sum(RS_scores) / len(RS_scores)))

                # VideoTitleFeature(BM25)
                out.write(" 14:" + str(np.sum(VN_scores) / len(VN_scores)))
                # outSERP.write(" 14:" + str(np.sum(VN_scores) / len(VN_scores)))

                # VideoDescriptionFeature(BM25)
                out.write(" 15:" + str(np.sum(VD_scores) / len(VD_scores)))
                # outSERP.write(" 15:" + str(np.sum(VD_scores) / len(VD_scores)))

                #NumberofOptionsFeature
                out.write(" 16:" + str(len(valu) - 1))
                # outNonSERP.write(" 1:" + str(len(valu) - 2))

                # QueryFeature(CosineSimilarityTFIDF)
                out.write(" 17:" + str(cosinetfidfQU_num))
                # outNonSERP.write(" 2:" + str(cosinetfidfQU_num))

                # QueryFeature(NumberofTermMatching)
                out.write(" 18:" + str(fu_QU))
                # outNonSERP.write(" 3:" + str(fu_QU))

                # QueryFeature(BM25)
                out.write(" 19:" + str(np.sum(QU_scores) / len(QU_scores)))
                # outNonSERP.write(" 4:" + str(np.sum(QU_scores) / len(QU_scores)))

                #StatementorQuestion
                if (valu[1].startswith("w")):
                    out.write(" 20:" + str(1))
                    # outNonSERP.write(" 5:" + str(1))
                else:
                    out.write(" 20:" + str(0))
                    # outNonSERP.write(" 5:" + str(0))

                #option1(writefeatures)
                #DocumentTitleFeature(CosineSimilarityTFIDF)
                out.write(" 21:" + str(cosinetfidfDT1_num))
                # outSERP.write(" 16:" + str(cosinetfidfDT1_num))

                # DocumentSnippetFeature(CosineSimilarityTFIDF)
                out.write(" 22:" + str(cosinetfidfSN1_num))
                # outSERP.write(" 17:" + str(cosinetfidfSN1_num))

                # RelatedSearchFeature(CosineSimilarityTFIDF)
                out.write(" 23:" + str(cosinetfidfRS1_num))
                # outSERP.write(" 18:" + str(cosinetfidfRS1_num))

                # VideoTitleFeature(CosineSimilarityTFIDF)
                out.write(" 24:" + str(cosinetfidfVN1_num))
                # outSERP.write(" 19:" + str(cosinetfidfVN1_num))

                # VideDescriptionFeature(CosineSimilarityTFIDF)
                out.write(" 25:" + str(cosinetfidfVD1_num))
                # outSERP.write(" 20:" + str(cosinetfidfVD1_num))

                # DocumentTitleFeature(NumberofTermMatching)
                out.write(" 26:" + str(fu_DT1))
                # outSERP.write(" 21:" + str(fu_DT1))

                # DocumentSnippetFeature(NumberofTermMatching)
                out.write(" 27:" + str(fu_SN1))
                # outSERP.write(" 22:" + str(fu_SN1))

                # RelatedSearchFeature(NumberofTermMatching)
                out.write(" 28:" + str(fu_RS1))
                # outSERP.write(" 23:" + str(fu_RS1))

                # VideoTitleFeature(NumberofTermMatching)
                out.write(" 29:" + str(fu_VN1))
                # outSERP.write(" 24:" + str(fu_VN1))

                # VideoDescriptionFeature(NumberofTermMatching)
                out.write(" 30:" + str(fu_VD1))
                # outSERP.write(" 25:" + str(fu_VD1))

                # DocumentTitleFeature(BM25)
                out.write(" 31:" + str(np.sum(DT_scores1) / len(DT_scores1)))
                # outSERP.write(" 26:" + str(np.sum(DT_scores1) / len(DT_scores1)))

                # DocumentSnippetFeature(BM25)
                out.write(" 32:" + str(np.sum(SN_scores1) / len(SN_scores1)))
                # outSERP.write(" 27:" + str(np.sum(SN_scores1) / len(SN_scores1)))

                # RelatedSearchFeature(BM25)
                out.write(" 33:" + str(np.sum(RS_scores1) / len(RS_scores1)))
                # outSERP.write(" 28:" + str(np.sum(RS_scores1) / len(RS_scores1)))

                # VideoTitleFeature(BM25)
                out.write(" 34:" + str(np.sum(VN_scores1) / len(VN_scores1)))
                # outSERP.write(" 29:" + str(np.sum(VN_scores1) / len(VN_scores1)))

                # VideoDescriptionFeature(BM25)
                out.write(" 35:" + str(np.sum(VD_scores1) / len(VD_scores1)))
                # outSERP.write(" 30:" + str(np.sum(VD_scores1) / len(VD_scores1)))

                # QueryFeature(CosineSimilarityTFIDF)
                out.write(" 36:" + str(cosinetfidfQU1_num))
                # outNonSERP.write(" 6:" + str(cosinetfidfQU1_num))

                # QueryFeature(NumberofTermMatching)
                out.write(" 37:" + str(fu_QU1))
                # outNonSERP.write(" 7:" + str(fu_QU1))

                # QueryFeature(BM25)
                out.write(" 38:" + str(np.sum(QU_scores1) / len(QU_scores1)))
                # outNonSERP.write(" 8:" + str(np.sum(QU_scores1) / len(QU_scores1)))

                #option2(writefeatures)
                # DocumentTitleFeature(CosineSimilarityTFIDF)
                out.write(" 39:" + str(cosinetfidfDT2_num))
                # outSERP.write(" 31:" + str(cosinetfidfDT2_num))

                # DocumentSnippetFeature(CosineSimilarityTFIDF)
                out.write(" 40:" + str(cosinetfidfSN2_num))
                # outSERP.write(" 32:" + str(cosinetfidfSN2_num))

                # RelatedSearchFeature(CosineSimilarityTFIDF)
                out.write(" 41:" + str(cosinetfidfRS2_num))
                # outSERP.write(" 33:" + str(cosinetfidfRS2_num))

                # VideoTitleFeature(CosineSimilarityTFIDF)
                out.write(" 42:" + str(cosinetfidfVN2_num))
                # outSERP.write(" 34:" + str(cosinetfidfVN2_num))

                # VideDescriptionFeature(CosineSimilarityTFIDF)
                out.write(" 43:" + str(cosinetfidfVD2_num))
                # outSERP.write(" 35:" + str(cosinetfidfVD2_num))

                # DocumentTitleFeature(NumberofTermMatching)
                out.write(" 44:" + str(fu_DT2))
                # outSERP.write(" 36:" + str(fu_DT2))

                # DocumentSnippetFeature(NumberofTermMatching)
                out.write(" 45:" + str(fu_SN2))
                # outSERP.write(" 37:" + str(fu_SN2))

                # RelatedSearchFeature(NumberofTermMatching)
                out.write(" 46:" + str(fu_RS2))
                # outSERP.write(" 38:" + str(fu_RS2))

                # VideoTitleFeature(NumberofTermMatching)
                out.write(" 47:" + str(fu_VN2))
                # outSERP.write(" 39:" + str(fu_VN2))

                # VideoDescriptionFeature(NumberofTermMatching)
                out.write(" 48:" + str(fu_VD2))
                # outSERP.write(" 40:" + str(fu_VD2))

                # DocumentTitleFeature(BM25)
                out.write(" 49:" + str(np.sum(DT_scores2) / len(DT_scores2)))
                # outSERP.write(" 41:" + str(np.sum(DT_scores2) / len(DT_scores2)))

                # DocumentSnippetFeature(BM25)
                out.write(" 50:" + str(np.sum(SN_scores2) / len(SN_scores2)))
                # outSERP.write(" 42:" + str(np.sum(SN_scores2) / len(SN_scores2)))

                # RelatedSearchFeature(BM25)
                out.write(" 51:" + str(np.sum(RS_scores2) / len(RS_scores2)))
                # outSERP.write(" 43:" + str(np.sum(RS_scores2) / len(RS_scores2)))

                # VideoTitleFeature(BM25)
                out.write(" 52:" + str(np.sum(VN_scores2) / len(VN_scores2)))
                # outSERP.write(" 44:" + str(np.sum(VN_scores2) / len(VN_scores2)))

                # VideoDescriptionFeature(BM25)
                out.write(" 53:" + str(np.sum(VD_scores2) / len(VD_scores2)))
                # outSERP.write(" 45:" + str(np.sum(VD_scores2) / len(VD_scores2)))

                # QueryFeature(CosineSimilarityTFIDF)
                out.write(" 54:" + str(cosinetfidfQU2_num))
                # outNonSERP.write(" 9:" + str(cosinetfidfQU2_num))

                # QueryFeature(NumberofTermMatching)
                out.write(" 55:" + str(fu_QU2))
                # outNonSERP.write(" 10:" + str(fu_QU2))

                # QueryFeature(BM25)
                out.write(" 56:" + str(np.sum(QU_scores2) / len(QU_scores2)))
                # outNonSERP.write(" 11:" + str(np.sum(QU_scores2) / len(QU_scores2)))

                #option3(writefeatures)
                # DocumentTitleFeature(CosineSimilarityTFIDF)
                out.write(" 57:" + str(cosinetfidfDT3_num))
                # outSERP.write(" 46:" + str(cosinetfidfDT3_num))

                # DocumentSnippetFeature(CosineSimilarityTFIDF)
                out.write(" 58:" + str(cosinetfidfSN3_num))
                # outSERP.write(" 47:" + str(cosinetfidfSN3_num))

                # RelatedSearchFeature(CosineSimilarityTFIDF)
                out.write(" 59:" + str(cosinetfidfRS3_num))
                # outSERP.write(" 48:" + str(cosinetfidfRS3_num))

                # VideoTitleFeature(CosineSimilarityTFIDF)
                out.write(" 60:" + str(cosinetfidfVN3_num))
                # outSERP.write(" 49:" + str(cosinetfidfVN3_num))

                # VideDescriptionFeature(CosineSimilarityTFIDF)
                out.write(" 61:" + str(cosinetfidfVD3_num))
                # outSERP.write(" 50:" + str(cosinetfidfVD3_num))

                # DocumentTitleFeature(NumberofTermMatching)
                out.write(" 62:" + str(fu_DT3))
                # outSERP.write(" 51:" + str(fu_DT3))

                # DocumentSnippetFeature(NumberofTermMatching)
                out.write(" 63:" + str(fu_SN3))
                # outSERP.write(" 52:" + str(fu_SN3))

                # RelatedSearchFeature(NumberofTermMatching)
                out.write(" 64:" + str(fu_RS3))
                # outSERP.write(" 53:" + str(fu_RS3))

                # VideoTitleFeature(NumberofTermMatching)
                out.write(" 65:" + str(fu_VN3))
                # outSERP.write(" 54:" + str(fu_VN3))

                # VideoDescriptionFeature(NumberofTermMatching)
                out.write(" 66:" + str(fu_VD3))
                # outSERP.write(" 55:" + str(fu_VD3))

                # DocumentTitleFeature(BM25)
                out.write(" 67:" + str(np.sum(DT_scores3) / len(DT_scores3)))
                # outSERP.write(" 56:" + str(np.sum(DT_scores3) / len(DT_scores3)))

                # DocumentSnippetFeature(BM25)
                out.write(" 68:" + str(np.sum(SN_scores3) / len(SN_scores3)))
                # outSERP.write(" 57:" + str(np.sum(SN_scores3) / len(SN_scores3)))

                # RelatedSearchFeature(BM25)
                out.write(" 69:" + str(np.sum(RS_scores3) / len(RS_scores3)))
                # outSERP.write(" 58:" + str(np.sum(RS_scores3) / len(RS_scores3)))

                # VideoTitleFeature(BM25)
                out.write(" 70:" + str(np.sum(VN_scores3) / len(VN_scores3)))
                # outSERP.write(" 59:" + str(np.sum(VN_scores3) / len(VN_scores3)))

                # VideoDescriptionFeature(BM25)
                out.write(" 71:" + str(np.sum(VD_scores3) / len(VD_scores3)))
                # outSERP.write(" 60:" + str(np.sum(VD_scores3) / len(VD_scores3)))

                # QueryFeature(CosineSimilarityTFIDF)
                out.write(" 72:" + str(cosinetfidfQU3_num))
                # outNonSERP.write(" 12:" + str(cosinetfidfQU3_num))

                # QueryFeature(NumberofTermMatching)
                out.write(" 73:" + str(fu_QU3))
                # outNonSERP.write(" 13:" + str(fu_QU3))

                # QueryFeature(BM25)
                out.write(" 74:" + str(np.sum(QU_scores3) / len(QU_scores3)))
                # outNonSERP.write(" 14:" + str(np.sum(QU_scores3) / len(QU_scores3)))

                #option4(writefeatures)
                # DocumentTitleFeature(CosineSimilarityTFIDF)
                out.write(" 75:" + str(cosinetfidfDT4_num))
                # outSERP.write(" 61:" + str(cosinetfidfDT4_num))

                # DocumentSnippetFeature(CosineSimilarityTFIDF)
                out.write(" 76:" + str(cosinetfidfSN4_num))
                # outSERP.write(" 62:" + str(cosinetfidfSN4_num))

                # RelatedSearchFeature(CosineSimilarityTFIDF)
                out.write(" 77:" + str(cosinetfidfRS4_num))
                # outSERP.write(" 63:" + str(cosinetfidfRS4_num))

                # VideoTitleFeature(CosineSimilarityTFIDF)
                out.write(" 78:" + str(cosinetfidfVN4_num))
                # outSERP.write(" 64:" + str(cosinetfidfVN4_num))

                # VideDescriptionFeature(CosineSimilarityTFIDF)
                out.write(" 79:" + str(cosinetfidfVD4_num))
                # outSERP.write(" 65:" + str(cosinetfidfVD4_num))

                # DocumentTitleFeature(NumberofTermMatching)
                out.write(" 80:" + str(fu_DT4))
                # outSERP.write(" 66:" + str(fu_DT4))

                # DocumentSnippetFeature(NumberofTermMatching)
                out.write(" 81:" + str(fu_SN4))
                # outSERP.write(" 67:" + str(fu_SN4))

                # RelatedSearchFeature(NumberofTermMatching)
                out.write(" 82:" + str(fu_RS4))
                # outSERP.write(" 68:" + str(fu_RS4))

                # VideoTitleFeature(NumberofTermMatching)
                out.write(" 83:" + str(fu_VN4))
                # outSERP.write(" 69:" + str(fu_VN4))

                # VideoDescriptionFeature(NumberofTermMatching)
                out.write(" 84:" + str(fu_VD4))
                # outSERP.write(" 70:" + str(fu_VD4))

                # DocumentTitleFeature(BM25)
                out.write(" 85:" + str(np.sum(DT_scores4) / len(DT_scores4)))
                # outSERP.write(" 71:" + str(np.sum(DT_scores4) / len(DT_scores4)))

                # DocumentSnippetFeature(BM25)
                out.write(" 86:" + str(np.sum(SN_scores4) / len(SN_scores4)))
                # outSERP.write(" 72:" + str(np.sum(SN_scores4) / len(SN_scores4)))

                # RelatedSearchFeature(BM25)
                out.write(" 87:" + str(np.sum(RS_scores4) / len(RS_scores4)))
                # outSERP.write(" 73:" + str(np.sum(RS_scores4) / len(RS_scores4)))

                # VideoTitleFeature(BM25)
                out.write(" 88:" + str(np.sum(VN_scores4) / len(VN_scores4)))
                # outSERP.write(" 74:" + str(np.sum(VN_scores4) / len(VN_scores4)))

                # VideoDescriptionFeature(BM25)
                out.write(" 89:" + str(np.sum(VD_scores4) / len(VD_scores4)))
                # outSERP.write(" 75:" + str(np.sum(VD_scores4) / len(VD_scores4)))

                # QueryFeature(CosineSimilarityTFIDF)
                out.write(" 90:" + str(cosinetfidfQU4_num))
                # outNonSERP.write(" 15:" + str(cosinetfidfQU4_num))

                # QueryFeature(NumberofTermMatching)
                out.write(" 91:" + str(fu_QU4))
                # outNonSERP.write(" 16:" + str(fu_QU4))

                # QueryFeature(BM25)
                out.write(" 92:" + str(np.sum(QU_scores4) / len(QU_scores4)))
                # outNonSERP.write(" 17:" + str(np.sum(QU_scores4) / len(QU_scores4)))

                #option5(writefeatures)
                # DocumentTitleFeature(CosineSimilarityTFIDF)
                out.write(" 93:" + str(cosinetfidfDT5_num))
                # outSERP.write(" 76:" + str(cosinetfidfDT5_num))

                # DocumentSnippetFeature(CosineSimilarityTFIDF)
                out.write(" 94:" + str(cosinetfidfSN5_num))
                # outSERP.write(" 77:" + str(cosinetfidfSN5_num))

                # RelatedSearchFeature(CosineSimilarityTFIDF)
                out.write(" 95:" + str(cosinetfidfRS5_num))
                # outSERP.write(" 78:" + str(cosinetfidfRS5_num))

                # VideoTitleFeature(CosineSimilarityTFIDF)
                out.write(" 96:" + str(cosinetfidfVN5_num))
                # outSERP.write(" 79:" + str(cosinetfidfVN5_num))

                # VideDescriptionFeature(CosineSimilarityTFIDF)
                out.write(" 97:" + str(cosinetfidfVD5_num))
                # outSERP.write(" 80:" + str(cosinetfidfVD5_num))

                # DocumentTitleFeature(NumberofTermMatching)
                out.write(" 98:" + str(fu_DT5))
                # outSERP.write(" 81:" + str(fu_DT5))

                # DocumentSnippetFeature(NumberofTermMatching)
                out.write(" 99:" + str(fu_SN5))
                # outSERP.write(" 82:" + str(fu_SN5))

                # RelatedSearchFeature(NumberofTermMatching)
                out.write(" 100:" + str(fu_RS5))
                # outSERP.write(" 83:" + str(fu_RS5))

                # VideoTitleFeature(NumberofTermMatching)
                out.write(" 101:" + str(fu_VN5))
                # outSERP.write(" 84:" + str(fu_VN5))

                # VideoDescriptionFeature(NumberofTermMatching)
                out.write(" 102:" + str(fu_VD5))
                # outSERP.write(" 85:" + str(fu_VD5))

                # DocumentTitleFeature(BM25)
                out.write(" 103:" + str(np.sum(DT_scores5) / len(DT_scores5)))
                # outSERP.write(" 86:" + str(np.sum(DT_scores5) / len(DT_scores5)))

                # DocumentSnippetFeature(BM25)
                out.write(" 104:" + str(np.sum(SN_scores5) / len(SN_scores5)))
                # outSERP.write(" 87:" + str(np.sum(SN_scores5) / len(SN_scores5)))

                # RelatedSearchFeature(BM25)
                out.write(" 105:" + str(np.sum(RS_scores5) / len(RS_scores5)))
                # outSERP.write(" 88:" + str(np.sum(RS_scores5) / len(RS_scores5)))

                # VideoTitleFeature(BM25)
                out.write(" 106:" + str(np.sum(VN_scores5) / len(VN_scores5)))
                # outSERP.write(" 89:" + str(np.sum(VN_scores5) / len(VN_scores5)))

                # VideoDescriptionFeature(BM25)
                out.write(" 107:" + str(np.sum(VD_scores5) / len(VD_scores5)))
                # outSERP.write(" 90:" + str(np.sum(VD_scores5) / len(VD_scores5)))

                # QueryFeature(CosineSimilarityTFIDF)
                out.write(" 108:" + str(cosinetfidfQU5_num))
                # outNonSERP.write(" 18:" + str(cosinetfidfQU5_num))

                # QueryFeature(NumberofTermMatching)
                out.write(" 109:" + str(fu_QU5))
                # outNonSERP.write(" 19:" + str(fu_QU5))

                # QueryFeature(BM25)
                out.write(" 110:" + str(np.sum(QU_scores5) / len(QU_scores5)))
                # outNonSERP.write(" 20:" + str(np.sum(QU_scores5) / len(QU_scores5)))

                out.write('\n')
                # outSERP.write('\n')
                # outNonSERP.write('\n')

    Video_Dict.clear()
    rsearch_Dict.clear()
    Webpage_Dict.clear()

f.close()
print("Done")
out.close()
