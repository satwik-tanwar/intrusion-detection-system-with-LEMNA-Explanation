# intrusion-detection-system-with-LEMNA-Explanation

I used the IoTID20 Dataset which is proposed in the paper ["A Scheme for Generating a Dataset for Anomalous Activity Detection in IoT Networks"](https://link.springer.com/chapter/10.1007/978-3-030-47358-7_52) which can be downloaded from [here](https://drive.google.com/file/d/1P2S4ErvWk-bEPF38n-sHJl7_KhXu6aKs/view?usp=share_link) to train a model which can detect an intrusion in a network.

Further, I used the LEMNA Algorithm which was proposed in the paper ["Lemna: Explaining deep learning based security applications"](https://gangw.cs.illinois.edu/ccs18.pdf) to generate explanations for the predictions of the model. LEMNA outputs the ranks of all the features for a particular instance. I also ran fidelity tests to test its performance against LIME. The results were in the favour of LEMNA.
