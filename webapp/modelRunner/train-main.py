# For training - main
import os
import tensorflow as tf

from tensorflow import keras
from inmodel import InpaintingModel
from augment import createAugment


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

traingen = createAugment(x_train, x_train)
testgen = createAugment(x_test, x_test, shuffle=False)

keras.backend.clear_session()
model = InpaintingModel().prepare_model()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[dice_coef])
# keras.utils.plot_model(model, show_shapes=True, dpi=76, to_file='model_v1.png')


if __name__ == "__main__":
    model.fit_generator(traingen, validation_data=testgen,
            epochs=10,
            steps_per_epoch=len(traingen),
            validation_steps=len(testgen),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='modelRunner/content/model-{epoch:02d}.h5')]
    )

    # model.save(os.path.join(os.path.dirname(__file__),'final_trained_model_test_main.h5'))

    model_json = model.to_json()
    with open("modelRunner/model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("modelRunner/model.h5")
    print("Saved model to disk")
