from utils import *

class JsonSimilarity:

    def __init__(self, apps_data):
        self.apps_data = apps_data

    def similarity_search(self, new_app_data):
        app_ids = list(self.apps_data.keys())
        search_values = [get_app_search_value(self.apps_data[app_id]["ai_dict"]) for app_id in app_ids]
        # search_values = [apps_data[app_id]["search_value"] for app_id in app_ids]
        new_app_id = list(new_app_data.keys())[0]
        search_values.append(get_app_search_value(new_app_data[new_app_id]["ai_dict"]))

        processed_docs = [preprocess_text(doc) for doc in search_values]
        tfs = [calculate_tf(doc) for doc in processed_docs]
        idf = calculate_idf(processed_docs)
        tfidfs = [calculate_tfidf(tf, idf) for tf in tfs]

        similarities = {}
        for i in range(len(tfidfs)-1):
            similarity = cosine_similarity(tfidfs[i], tfidfs[-1])
            similarities[app_ids[i]] = round(similarity, 4)

        highest_similarity_app = max(similarities, key=similarities.get)
        final = {highest_similarity_app: self.apps_data[highest_similarity_app]}
        return final
