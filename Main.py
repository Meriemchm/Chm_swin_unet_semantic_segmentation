if __name__ == '__main__':
    from ConfigFile import config
    from Architecture2 import SwinUnet
    from ModelTrain import trainer

    net = SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES).cuda()
    output_dir = "./output_7"
    trainer(config, net, output_dir)
