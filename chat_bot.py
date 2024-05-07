from flask import *
import json
import string
import random
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk import word_tokenize
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download('omw-1.4')
# from googletrans import Translator

app = Flask(__name__)

#adding questions and answers to the bot
# data = {"intents": [
#              {"tag": "Nutrition1",
#               "patterns": ["What are the foods to be avoided during cancer ?"],
#               "responses": ["During the treatment of cancer, avoid raw foods, uncovered outside food,processed foods, canteen food, thin skin and uneven skin fruits."],
#              },
#              {"tag": "Nutrition2",
#               "patterns": ["Should thin-skinned fruits be avoided for life?"],
#               "responses": ["No, thin skin fruits can be consumed after completion of the cancer treatment."]
#              },
#              {"tag": "Nutrition3",
#               "patterns": ["what are the foods to be taken to improve the blood?"],
#               "responses": ["Iron rich foods like green leafy vegetables, well cooked eggs, gardencress seeds, dates and raisins in cooked form can be consumed along with complete balanced diet."]
#              },
#              {"tag": "Nutrition4",
#               "patterns": ["can we consume eggs during fever?"],
#               "responses": ["Yes, well cooked eggs can be consumed during fever."]
#              },
#              {"tag": "Treatment Related1",
#               "patterns": [ "can we give red meat or organ meat during the treatment?"],
#               "responses": ["No, as it may possibly introduce infection to the child during the treatment phase."]
#              },
#             {"tag":"Treatment Related2",
#              "patterns": ["can we feed the child during the chemotherapy infusion/ blood infusion?"],
#              "responses": ["Yes, child can be fed during the chemotherapy or blood infusion."]
#              },
#             {"tag":"Nutrition5",
#              "patterns":["why  kiwi and papaya juice shouldn't be consumed during the treatment, as it is said that they improve blood and Platelet count?"],
#              "responses":["Kiwi and Papaya juice alone cannot improve Platelet Count in cancer and related intensive treatments. Eating it raw, increases the risk of infection and therefore would compromise the nutritional needs of the child."]
#             },
#             {"tag":"Treatment Related3",
#              "patterns":["Can sour food items be eaten during the treatment ?"],
#              "responses":["Yes, sour food items can be eaten during the treatment as per tolerance."]
#             },
#             {"tag":"Nutrition6",
#             "patterns":["Can milk be consumed during cough and cold ?"],
#             "responses":["Yes, milk with turmeric is a good option to fight cough and cold."]
#             },
#             {"tag":"Nutrition7",
#             "patterns":["Can Apple be consumed ?"],
#             "responses":["Apple is a thin skin fruit which means it is susceptible to pest and microrganism infestation, therefore it is avoided when the blood count is low. Otherwise It can be consumed in a well washed, peeled or cooked form."]
#             },
#             {"tag":"Treatment Related5",
#             "patterns":["Is it ok to have maida products during treatment?"],
#             "responses":["It is better to be avoided as it lacks fiber and nutritients."]
#             },
#             {"tag":"Treatment Related6",
#             "patterns":["is it ok to have pickles during treatment?"],
#             "responses":["Salted pickles contain high amounts of salt and oil and holds no nutrition value. Pickles are raw and holds risk of infection, therefore to be avoided during treatment."]
#             },
#             {"tag":"Treatment Related7",
#             "patterns":["Can one eat eggplant, potato, gogu leaves during treatment ?"],
#             "responses":["Yes, these foods can be consumed in well cooked form along with a well balanced diet."]
#             },
#             {"tag":"Treatment Related8",
#             "patterns":["Can yogurt be eaten during treatment?"],
#             "responses":["Yes, curd is full of good proteins and probiotics."]
#             },
#             {"tag":"Treatment Related9",
#             "patterns":["can we feed kiwi & dragon fruit daily to increase platelets?"],
#             "responses":["Kiwi and Dragon Fruit cannot suffice the need of improving platelets in an intensive treatment."]
#             },
#             {"tag":"Nutrition8",
#             "patterns":["Is it okay to have 'Boost' or 'Horlics' sometimes?"],
#             "responses":["It can be avoided, as it contains sugar and is low on nutrition value. However a well balanced meal provides holistic nutriton."]
#             },
#             {"tag":"Nutrition9",
#             "patterns":["can we add ghee or milk if he is coughing?"],
#             "responses":["Yes, as ghee and milk does not aggreviate cough, milk with turmeric is a good option to fight cough, as it is antibacterial, antiseptic properties.   "]
#             },
#             {"tag":"Nutrition10",
#             "patterns":["should we avoid milk during cough"],
#             "responses":["No, milk with turmeric is a good option to fight cough."]
#             },
#             {"tag":"Treatment Related10",
#             "patterns":["Can patient have raw Cucumber?"],
#             "responses":["No, raw foods to be avoided during the treatment phase. uncooked food holds risk of infection"]
#             },
#             {"tag":"Nutrition11",
#             "patterns":["Is it ok to have ice cream sometimes ?"],
#             "responses":["Yes, sometimes it is okay to consume icecream which is well packed and sealed on manufacturing from a good brand."]
#             },
#             {"tag":"Nutrition12",
#             "patterns":["Does giving beetroot and fruits increase blood ?"],
#             "responses":["No, It does not alone increase blood, eating a whole fruit and cooked beetroot along with a balance diet is a better choice."]
#             },
#             {"tag":"Nutrition13",
#             "patterns":["What foods help in increasing platelets ?"],
#             "responses":["There are no specific foods to increase platelets, a well balanced diet is the only holistic approach."]
#             },
#             {"tag":"Nutrition14",
#             "patterns":["Is it okay to have 'lays chips'?"],
#             "responses":["No, it is high in salts, unhealthy fats, preservative and hold no nutrition value."]
#             },
#             {"tag":"Nutrition15",
#             "patterns":["Is it ok to give maggi during treatment?"],
#             "responses":["No, it is high in salts, unhealthy fats, preservative and hold no nutrition value."]
#             },
#             {"tag":"Nutrition16",
#             "patterns":["Is it okay to have almonds during chemotherapy?"],
#             "responses":["Almonds can be soaked or roasted and consumed during chemotherapy."]
#             },
#             {"tag":"Nutrition17",
#             "patterns":["Can I give beet root juice, carrot juice?"],
#             "responses":["No, eating well cooked whole beetroot and carrots is a better option."]
#             },
#             {"tag":"Nutrition18",
#             "patterns":["Can non veg soup be consumed, next day of chemotherapy ?"],
#             "responses":["Yes, home cooked chicken soup, after proper cleaning and handling of raw meat."]
#             },
#             {"tag":"Nutrition19",
#             "patterns":["Can we give act 2 popcorn to my child?"],
#             "responses":["No, instead home made popcorn are healthy and has no unhealthy fats."]
#             },
#             {"tag":"Nutrition20",
#             "patterns":["My child is having semi solid stool after eating green leaves. Should I avoid adding green leaves?"],
#             "responses":["The reasons for semi solid stool can be numerous, the doctor and dietitians can help identify the cause. Well cleaned, finely chopped and well cooked green leafy vegetable in an advisable quantity can be consumed."]
#             },
#             {"tag":"Nutrition21",
#             "patterns":["My child never like the smell of ghee even if added in any other form of food. What to do?"],
#             "responses":["Add ghee while kneading chapati/roti dough or can add ghee with oil while cooking food."]
#             },
#             {"tag":"Nutrition22",
#             "patterns":["My child is not drinking plain milk, what can be done ?"],
#             "responses":["Milkshake can be made using fruits, nuts, unsweetened cocoa powder to add flavour to the milk. Give milk in form of curd, lassi, paneer, buttermilk, sevaiya, kheer, halva or add while kneading wheat dough."]
#             },
#             {"tag":"Nutrition23",
#             "patterns":["Can we give Honey or Chyawanprash to the child ?"],
#             "responses":["Honey and Chyawanprash are raw in nature, they hold risk of infection, it can be avoided. A well balanced diet is best source of nutrition."]
#             },
#             {"tag":"Nutrition24",
#             "patterns":["Is it necessary to boil RO water?"],
#             "responses":["Yes, any form of water is to be boiled and consumed to avoid infecton."]
#             },
#             {"tag":"Nutrition25",
#             "patterns":["which vegetable are not allowed during treatment?"],
#             "responses":["All well cooked vegetables are allowed."]
#             },
#             {"tag":"Nutrition26",
#             "patterns":["my child is not eating egg yolk, what to do?"],
#             "responses":["Egg yolk can be mixed in the meals (khichdi/dal/vegetable curry/omlette/scrambled eggs) to camoflauge the taste and be consumed."]
#             },
#             {"tag":"Nutrition27",
#             "patterns":["How often can we give bread ?"],
#             "responses":["Brown bread can be given in a situation when home cooked food is not available while travelling or hospital stay. Home cooked chapati, bhakri, hot breakfast items or snacks should always be the first choice "]
#             },
#             {"tag":"Nutrition28",
#             "patterns":["Can soaked chana be given, as it has more protein than cooked ones?"],
#             "responses":["Cooked Chana has equal amount of protein as the raw one. Cooked chana is better digestible and has lesser chance of infection"]
#             },
#             {"tag":"Treatment Related11",
#             "patterns":["Can coconut water be consumed everyday?"],
#             "responses":["Coconut water should be avoided during the induction phase of treatment. However, in the other treatment phases it can be consumed 2 to 3 times a week along with other liquids. "]
#             },
#             {"tag":"Nutrition29",
#             "patterns":["Can we continue with the nutritional supplement even stopped by nutritionist, as it is wholesome nutrition ?"],
#             "responses":["Supplement is consumed as an adjuvant to a wholesome balanced diet, during the period of increased nutritional needs. Post treatment home cooked food should be encouraged as a sustainable option."]
#             },
#             {"tag":"Treatment Related12",
#             "patterns":["My child has loose motions what should I give ?"],
#             "responses":["In diarrhea one should avoid milk, instead curd, buttermilk are better for digestion. Fully cooked, soft foods like khichdi, porridges, curd-rice, banana, apple along with plenty of fluids and ORS should be consumed throughout the day. Refer to the video for more details. "]
#             },
#             {"tag":"Nutrition30",
#             "patterns":["Can I use the same supplements given by Cuddles for other children in the family ?"],
#             "responses":["The supplement is recomended keeping in mind your child's clinical condition, current intake and the increased nutritional requirements. The supplement is issued in a calculated form and hence using the same for another child may lead to underfeeding and delayed improvement in the nutritional status. "]
#             },
#             {"tag":"Only",
#             "patterns":["Hi"],
#             "responses":["Hi my name is Bot how may I help you."]
#             }

# ]}

# #lemmatization and tokenization
# lemmatizer = WordNetLemmatizer()

# words = []
# classes = []
# doc_x = []
# doc_y = []

# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         tokens = word_tokenize(pattern)
#         words.extend(tokens)
#         doc_x.append(pattern)
#         doc_y.append(intent["tag"])
#     if intent["tag"] not in classes:
#         classes.append(intent["tag"])

# #bot training using deep learning
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# words = sorted(set(words))
# classes = sorted(set(classes))

# training = []
# out_empty = [0] * len(classes)

# for idx,doc in enumerate(doc_x):
#     bow = []
#     text = lemmatizer.lemmatize(doc.lower())
#     for word in words:
#         bow.append(1) if word in text else bow.append(0)
#     output_row = list(out_empty)
#     output_row[classes.index(doc_y[idx])] = 1
#     training.append([bow, output_row])

# random.shuffle(training)
# training = np.array(training, dtype=object)
# train_x = np.array(list(training[:,0]))
# train_y = np.array(list(training[:,1]))

# input_shape = (len(train_x[0]),)
# output_shape = len(train_y[0])
# epochs = 200

# model = Sequential()
# model.add(Dense(128, input_shape = input_shape, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.3))
# model.add(Dense(output_shape, activation = "softmax"))

# adam = tf.keras.optimizers.Adam(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=["accuracy"])
# model.fit(x=train_x,y=train_y,epochs=100,verbose=1)

# def clean_text(text):
#     tokens = nltk.word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return tokens

# def bag_of_words(text, vocab):
#     tokens = clean_text(text)
#     bow = [0] * len(vocab)
#     for w in tokens:
#         for idx, word in enumerate(vocab):
#             if word == w:
#                 bow[idx] = 1
#     return np.array(bow)

# def pred_class(text, vocab, labels):
#     bow = bag_of_words(text, vocab)
#     result = model.predict(np.array([bow]))[0]
#     thresh = 0.2
#     y_pred = [[idx,res] for idx, res in enumerate(result) if res > thresh]
#     y_pred.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in y_pred:
#         return_list.append(labels[r[0]])
#     return return_list

# def get_response(intents_list, intents_json):
#     tag = intents_list[0]
#     list_of_intents = intents_json["intents"]
#     for i in list_of_intents:
#         if i["tag"] == tag:
#             result = random.choice(i["responses"])
#             break
#     return result

# def ask_bot(message_eng):
#     intents = pred_class(message_eng, words, classes)
#     result = get_response(intents, data)
#     return result


@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.json.get('user_message')  # Access JSON data from the request
        if user_message:
            bot_response = "Kana kala se"  # Generate bot response based on user message
            return jsonify({"bot_response": bot_response})  # Return JSON response with bot response
        else:
            return jsonify({"error": "User message not found"}), 400  # Return error response if user message is not found

    return render_template('chat_bot.html')
    

if __name__ =='__main__':
    app.run(debug=True)