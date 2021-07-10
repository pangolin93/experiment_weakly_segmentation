import segmentation_models_pytorch as smp

ENCODER_NAME='efficientnet-b0'
ENCODER_WEIGHTS="imagenet"
ACTIVATION = 'softmax2d'

def get_segm_model_and_preprocess_fn():

    model = smp.FPN(
        encoder_name=ENCODER_NAME,  
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,                    # RGB
        classes=5,                        # model output channels
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_NAME, ENCODER_WEIGHTS)

    return model, preprocessing_fn
