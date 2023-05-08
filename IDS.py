# %% [markdown]
# # Imports

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LEMNA import lemna
import lime

# %%
df = pd.read_csv("IoTID20.csv")
print("Number of Columns: ", len(df.columns))

# %% [markdown]
# # Exploratory Analysis and Data Preprocessing

# %%
df = pd.read_csv("IoTID20.csv")
# print("Number of Columns: ", len(df.columns))

categories = df["Cat"].unique()
df.drop(df[df["Flow_Byts/s"] == float("inf")].index, inplace=True)

drop = [
    "Fwd_PSH_Flags",
    "Fwd_URG_Flags",
    "Fwd_Byts/b_Avg",
    "Fwd_Pkts/b_Avg",
    "Fwd_Blk_Rate_Avg",
    "Bwd_Byts/b_Avg",
    "Bwd_Pkts/b_Avg",
    "Bwd_Blk_Rate_Avg",
    "Init_Fwd_Win_Byts",
    "Fwd_Seg_Size_Min",
    "Cat",
    "Label",
]

df.drop(drop, axis=1, inplace=True)

from datetime import datetime


def get_timestamp(str):
    date_format = "%d/%m/%Y %I:%M:%S %p"
    timestamp = datetime.strptime(str, date_format).time()
    timestamp = (
        float(timestamp.hour)
        + float(timestamp.minute) / 60
        + float(timestamp.second) / 3600
    )
    return timestamp


df["Timestamp"] = df["Timestamp"].apply(lambda str: get_timestamp(str))

# %%
print("Number of Columns: ", len(df.columns))
print(df.columns)

# %%
df.head()

# %%
# 'Active_Max', 'Bwd_IAT_Max', 'Bwd_Seg_Size_Avg', 'Fwd_IAT_Max', 'Fwd_Seg_Size_Avg',
# 'Idle_Max', 'PSH_Flag_Cnt', 'Pkt_Size_Avg', 'Subflow_Bwd_Byts', 'Subflow_Bwd_Pkts',
# 'Subflow_Fwd_Byts', 'Subflow_Fwd_Pkts'

# %%
df.info()

# %%
class_names = df["Sub_Cat"].unique()
class_names

# %% [markdown]
# # Data preperation

# %%
cat_atr = [
    "Flow_ID",
    "Src_IP",
    "Src_Port",
    "Dst_IP",
    "Dst_Port",
    "Protocol",
    "Bwd_PSH_Flags",
    "Bwd_URG_Flags",
    "FIN_Flag_Cnt",
    "SYN_Flag_Cnt",
    "RST_Flag_Cnt",
    "PSH_Flag_Cnt",
    "ACK_Flag_Cnt",
    "URG_Flag_Cnt",
    "CWE_Flag_Count",
    "ECE_Flag_Cnt",
]
num_atr = []
for x in df.columns:
    if x in ["Sub_Cat"]:
        continue
    if x not in cat_atr:
        num_atr.append(x)

# %%
X = df.drop(["Sub_Cat"], axis=1)
y = df["Sub_Cat"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Validation Samples:", len(X_validation))


# %%
def prepare_input_data(df, cat_atr, encoder):
    scaler = StandardScaler()

    cat_data = encoder.transform(df[cat_atr])
    cat_data = pd.DataFrame(cat_data, columns=cat_atr)
    cat_data = np.asarray(cat_data).astype(np.int32)

    num_data = df.drop(cat_atr, axis=1)
    num_data = np.asarray(num_data).astype(np.float32)
    num_data = scaler.fit_transform(num_data)

    return (num_data, cat_data)


def prepare_labels(data, target_cols):
    labels = np.zeros((len(data), 9))
    i = 0
    for x in data:
        labels[i][np.where(target_cols == x)[0][0]] = 1
        i += 1
    return labels


# %%
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(df[cat_atr])
(X_train_num, X_train_cat) = prepare_input_data(X_train, cat_atr, encoder)
(X_validation_num, X_validation_cat) = prepare_input_data(
    X_validation, cat_atr, encoder
)
(X_test_num, X_test_cat) = prepare_input_data(X_test, cat_atr, encoder)

# %%
target_cols = df["Sub_Cat"].unique()
y_train = prepare_labels(y_train, target_cols)
y_validation = prepare_labels(y_validation, target_cols)
y_test = prepare_labels(y_test, target_cols)

# %%
n = 0
for x in encoder.categories_:
    n += len(x)

# %% [markdown]
# # Model


# %%
def create_model():
    # cat input
    inp_cat = tf.keras.layers.Input(
        shape=np.shape(X_train_cat)[1], name="categorical_input"
    )
    embedded = tf.keras.layers.Embedding(
        input_dim=n,
        output_dim=32,
        input_length=np.shape(X_train_cat)[1],
        name="categorical_embedding",
    )(inp_cat)
    flattened = tf.keras.layers.Flatten(name="flatten")(embedded)

    # num input
    inp_num = tf.keras.layers.Input(shape=len(num_atr), name="numerical_input")

    concatenated = tf.keras.layers.Concatenate(name="concatenation")(
        [inp_num, flattened]
    )

    # hidden layers
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(concatenated)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)
    hidden = keras.layers.Dense(
        1024, activation="selu", kernel_initializer="lecun_normal"
    )(hidden)

    # output
    output = keras.layers.Dense(9, activation="softmax", name="Output")(hidden)

    model = keras.Model(
        inputs=[inp_cat, inp_num],
        outputs=[output],
    )
    loss = "categorical_crossentropy"
    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4)
    metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# %%
# model=create_model()
# model.summary()

# %%
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
# checkpoint_cb = keras.callbacks.ModelCheckpoint('/content/drive/My Drive/ISM/classifier.h5',save_best_only=True)
# tr_history=model.fit([X_train_cat,X_train_num],y_train,validation_data=([X_validation_cat,X_validation_num],y_validation),
#           epochs=50,callbacks=[early_stopping_cb,checkpoint_cb])

# %%
# pd.DataFrame(tr_history.history).plot(figsize=(12,8))
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()

# %%
model = create_model()
model.summary()

# %%
model.load_weights("classifier.h5")
# model_evaluation=model.evaluate([X_test_cat,X_test_num],y_test)

# %%
# import statistics
# print("Loss: ", model_evaluation[0])
# print("Accuracy Score: ", model_evaluation[1])
# print("Precision Score: ", model_evaluation[2])
# print("Recall Score: ", model_evaluation[3])
# print("F1 Score: ", statistics.harmonic_mean([model_evaluation[2],model_evaluation[3]]))

# %% [markdown]
# # Explanations

# %%
training_df = np.array(
    pd.concat([pd.DataFrame(X_train_cat), pd.DataFrame(X_train_num)], axis=1)
)

# %%
test_df = np.array(
    pd.concat([pd.DataFrame(X_test_cat), pd.DataFrame(X_test_num)], axis=1)
)
feature_names = cat_atr + num_atr


def predFun(model, data):
    cat_data = data[:, 0 : len(cat_atr)]
    num_data = data[:, len(cat_atr) :]
    return model.predict([cat_data, num_data])


# %%
def explain_sample(i):
    exp = lemna.LemnaExplainer(model, test_df, feature_names, class_names, predFun)
    rankedFeatures = exp.extract_feature(test_df[i], 1000)
    rankedFeatures = pd.DataFrame(rankedFeatures)

    print("Prediction -->", class_names[np.argmax(exp.pred)])
    print("True --->", class_names[np.argmax(y_test[i])])

    print(rankedFeatures[:20])


# %%
explain_sample(1781)

# %% [markdown]
# ## Fidelity Tests

# %%
from LEMNA.FidelityTest import fid_test
import concurrent.futures


# %%
def predFun2(data, model=model):
    cat_data = data[:, 0 : len(cat_atr)]
    num_data = data[:, len(cat_atr) :]
    return model.predict([cat_data, num_data])


# fid tests for lime
def lime_feature_deduction_test(test_data, selected_fea, label_index):
    test_data = test_data.reshape(1, -1)
    test_data[:, selected_fea] = 0
    pred = predFun2(test_data)[:, label_index]
    return pred


def lime_synthetic_feature_test(test_data, selected_fea, label_index):
    test_data = test_data.reshape(1, -1)
    data = np.zeros((1, test_data.shape[1]))
    data[:, selected_fea] = test_data[:, selected_fea]
    pred = predFun2(data)[:, label_index]
    return pred


def lime_feature_augmentation_test(test_data, selected_fea, label_index):
    test_data = test_data.reshape(1, -1)
    random_index = np.random.choice(len(test_df), size=1, replace=False)
    test_seed = test_df[random_index]
    data = np.array(test_seed).reshape(1, -1)
    data[:, selected_fea] = test_data.reshape(1, -1)[:, selected_fea]
    pred = predFun2(data)[:, label_index]
    return pred


# %%
def run_tests_in_parallel(test_func, args_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(test_func, i) for i in args_list]
        concurrent.futures.wait(futures)


# %%
# Parameters for Fidelity test
n = 0
test1 = 0
test2 = 0
test3 = 0

test1_lime = 0
test2_lime = 0
test3_lime = 0

fea_num = 20
num_samples = range(25)

# %%
# Fidelity Tests

exp = lemna.LemnaExplainer(model, test_df, feature_names, class_names, predFun)
l = lime.lime_tabular.LimeTabularExplainer(
    test_df, class_names=class_names, feature_names=feature_names
)


def test(i):
    global n, test1, test2, test3, test1_lime, test2_lime, test3_lime
    orignal_pred = predFun(model, test_df[i].reshape(1, -1))
    label_index = np.argmax(orignal_pred)
    if orignal_pred[0, label_index] >= 0.5:
        n += 1
        rankedFeatures = exp.extract_feature(test_df[i], 1000)
        test = fid_test(exp)

        # Lemna fidelity tests
        fea_ded = test.feature_deduction_test(fea_num)
        if fea_ded >= 0.5:
            test1 += 1

        syn_fea = test.synthetic_feature_test(fea_num)
        if syn_fea >= 0.5:
            test2 += 1

        fea_aug = test.feature_augmentation_test(fea_num)
        if fea_aug >= 0.5:
            test3 += 1

        # Lime fidelity tests
        fea_lime = l.explain_instance(
            test_df[i], predFun2, num_features=fea_num
        ).local_exp
        fea_lime = np.array(fea_lime[1])[:, 0].astype(np.int32)

        fea_ded_lime = lime_feature_deduction_test(test_df[i], fea_lime, label_index)
        if fea_ded_lime >= 0.5:
            test1_lime += 1

        syn_fea_lime = lime_synthetic_feature_test(test_df[i], fea_lime, label_index)
        if syn_fea_lime >= 0.5:
            test2_lime += 1

        fea_aug_lime = lime_feature_augmentation_test(test_df[i], fea_lime, label_index)
        if fea_aug_lime >= 0.5:
            test3_lime += 1


# %%
run_tests_in_parallel(test, num_samples)

# %%
print("LEMNA Fidelity Tests Results")
print("Feature deduction test (value should be low) -->", test1 / n)
print("Synthetic feature test (value should be high) -->", test2 / n)
print("Feature Augmentation test (value should be high) -->", test3 / n)

print("\nLIME Fidelity Tests Results")
print("Feature deduction test (value should be low) -->", test1_lime / n)
print("Synthetic feature test (value should be high) -->", test2_lime / n)
print("Feature Augmentation test (value should be high) -->", test3_lime / n)

# %%
