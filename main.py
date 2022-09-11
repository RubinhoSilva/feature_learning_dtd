import click

import feature_learning


@click.command()
@click.option(
    '--architecture',
    '-a',
    type=click.Choice([
        'Xception',
        'VGG16',
        'VGG19',
        'ResNet50',
        'ResNet50',
        'InceptionV3',
        'InceptionResNetV2',
        'MobileNet',
        'MobileNetV2',
        'DenseNet121',
        'DenseNet201',
        'NASNetMobile',
        'NASNetLarge',
        'EfficientNetV2L',
        'ResNet50',
    ]),
    help='Which architecture will be used to extract the features',
    required=True
)
@click.option(
    '--path',
    '-p',
    default='dtd/images/',
    help='Path where the images are found',
)
@click.option(
    '--seed',
    '-s',
    default=1994,
    help='',
    type=int
)
@click.option(
    '--height_image',
    '-hi',
    default=300,
    help='',
    type=int
)
@click.option(
    '--width_image',
    '-wi',
    default=300,
    help='',
    type=int
)
@click.option(
    '--quantity_images_batch',
    '-qib',
    default=30,
    help='',
    type=int
)
@click.option(
    '--patience',
    default=3,
    help='',
    type=int
)
@click.option(
    '--epochs',
    '-e',
    default=25,
    help='',
    type=int
)
@click.option(
    '--learning_rate',
    '-lr',
    default=1e-04,
    help='',
    type=int
)
def main(architecture, path, seed, height_image, width_image, quantity_images_batch, patience, epochs, learning_rate):
    print("Architecture: " % architecture)

    feature_learning.main(path, seed, architecture, height_image, width_image, quantity_images_batch, patience,
                          epochs, learning_rate)

    # extract_features(model, patches, folds, height, width,
    #                  input_file_format_string, output_file_format_string, gpuid)


if __name__ == "__main__":
    main()
