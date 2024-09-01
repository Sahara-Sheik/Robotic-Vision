from settings import read_yaml
#, write_yaml

def main():
    print("Preprocess representation")
    print("Not implemented yet: specify the dataset (as a set of directories)") 
    print("Not implemented yet: run the representation and save the values") 

    #conf = read_json("/home/lboloni/Documents/Hackingwork/_Checkouts/VisionBasedRobotManipulator/src/settings-tredy2.json")
    
    conf = read_yaml("/home/lboloni/Documents/Hackingwork/_Checkouts/VisionBasedRobotManipulator/src/settings-tredy2.yaml")
    print(conf)
    print(conf["robot"])
    print(conf["robot"]["usb_port"])
    print(conf["conv_vae"]["model_dir"])

if __name__ == "__main__":
    main()

