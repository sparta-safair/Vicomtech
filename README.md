### Adversarial Machine Learning Benchmark Tool

This tool is intended to facilitate easy conversion of models trained using PyTorch or any other Deep Learning framework into
[FoolBox](https://github.com/bethgelab/foolbox). Foolbox is an open source framework which maintains regularly updated code implementation of latest
adversarial attacks. We have created this tool as a wrapper to Foolbox with functionality relevant to various use-caseses. 

This benchmark tool would allow an easy conversion of their PyTorch/Tensorflow/Keras etc models and hence, one can easily test his/her model
against all the attack implementations available.


### Tool Structure

We have structured our tool as follows:-
1. attacks - The module contains some sample attack implementations. One can use these as a template and easily implement other Foolbox adversarial attacks too.
2. dataset - Contains the code for loading differnt dataset such as[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
3. model - Contains the source code for a model. We have a PyTorch model in the directory as an example. Please note that we expect that the model is already trained and we are going to test it against adversarial.
perturbations. Hence, we ***do not*** provide any code for training of the model. For our example mode, in case you want to train a model from scratch, please check [here](https://git.sec.in.tum.de/Norouzian/safair-ai-contest).
4. use_cases - Contains logic specific to the  various use-casess. Our examples are Face-Reidentification and Attribute Alterations as outlined [here](https://www.sec.in.tum.de/i20/projects/sparta-safair-ai-contest).
5. wrapper - Contains the model converter. This would take a model and convert it into FoolBox.

We would like to re-iterate that we expect that the model is already trained. Hence, please make sure you have saved the model weights. The model conversion process
would first load the model weights and then convert the model to Foolbox.

### Execution Steps

To start with execute 
>>> python main.py -h

To get a list of options and instructions to execute the program.
For starting a simple training loop, simply use
>>> python main.py --task_type attr --checkpoint_dir saved_models --model_number 0

Since the options have default arguments in most of the cases (please check `main.py` file for the entire list), one can make use of the default options and
reduce the above command command to

>>> python main.py 

In our examples, we support 3 defenses currently.
1. AdversarialTrain
2. DimensionalityReduction (Top)
3. DimensionalityReduction (Middle)


This can be selected by using `defense_type` argument. For instance,
>>> python main.py --checkpoint_dir saved_models --defense_type DimensionalityReduction --model_number 0

Here we specify the `checkpoint_dir` and `model_number` (Specific model to load). We expect that model weights are present in the `saved_models` folder before starting the conversion process.

The results shall be computed on the CelebA dataset.


### Creating a new Attack

To create a new Attack, please create a new python file in the `attacks/attack_types` package. Please make sure all attacks extend the ```attacks.base.Attack``` class.

The class has three methods that are used:-
* instantiate_attack() -> Which is used in order to create an instance of `attack` type defined in Foolbox framework.
* attack_description() -> A string representation of the attack. Please check [this](#Config) to see how this comes in handy for the tool. 

Once this is done, the tool would automatically recognize the new Attack and compute the model performance against the new attack along with the previous ones (***Basically all the Attacks that are present in the `attacks.attack_types` package***)

However, there is a way to skip certain attacks if there is a need. Please check [here](#Config)

### Config

There may be scenarios in which you want to skip certain attack types (for instacne Carlini and Wagner). This can be done by editing the `config.py` file. The `skip_list` can be used to skip attacks.
The tool performs string matching based on the string representation of each attack class. For instance, if you want to skip c&w attack, just use
```
skip_list = ['carlini_wagner']
```   
The string representation of Carlini and Wagner attack is `carlini_wagner` and the same name needs to be used in the configuration list.

If the `skip_list` is empty, we run the model against all the test types.

