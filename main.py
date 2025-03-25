from training.pipeline import CustomPipeline
# from training.utils import *
import pickle

if __name__ == '__main__':
    log_pipe = CustomPipeline("logreg")
    log_pipe.preprocessing("physionet.org/files/widsdatathon2020/1.0.0/data/training_v2.csv",
                           "physionet.org/files/widsdatathon2020/1.0.0/data/WiDS_Datathon_2020_Dictionary.csv",
                           "hospital_death")

    # log_pipe.nested_cross_validation()  # Comment out if a model already exists

    with open('training/FINAL_model.pkl', 'rb') as file:
        log_pipe.model = pickle.load(file)
        print(log_pipe.model.get_params())
    log_pipe.train(apply_reweighting=False, calibrate=True)
    log_pipe.predict()
    log_pipe.eval()
    log_pipe.explain_model()

