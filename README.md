# deep-learning-go
deep-learning-go is util for github.com/patrikeh/go-deep
## install
```
$ go get https://github.com/ryomak/deep-learning-go
```
## example

```go
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	iclass "github.com/ryomak/deep-learning-go/image/image_classifier"
	util "github.com/ryomak/deep-learning-go/util"
)

func main() {
	i := iclass.Init(
		[]string{
			"tida",
			"yuna",
			"lulu",
		},
		"dataset",
		30,
		30,
	)
	rand.Seed(time.Now().UnixNano())
	
	data, err := i.MakePattern()
	if err != nil {
		panic(err)
	}
	
	ex := training.Examples(util.DatsetToDataSets(data))
	ex.Shuffle()

	neural, err := LoadModel("model.json")
	if err != nil {
		fmt.Println(err)
		neural = deep.NewNeural(&deep.Config{
			Inputs:     len(data[0].Input),
			Layout:     append([]int{1000,100}, len(data[0].Response)),
			Activation: deep.ActivationSoftmax,
			Mode:       deep.ModeMultiClass,
			Weight:     deep.NewNormal(1, 0),
			Bias:       true,
		})
		fmt.Println("train start ")
		fmt.Println(len(data))
		trainer := training.NewBatchTrainer(training.NewAdam(0.001, 0, 0, 0), 40, len(ex)/2, 12)

		trainer.Train(neural, ex, ex, 100)
		SaveModel(neural, "model.json")
	}
	
	gazou, err := i.Decode("input.jpg")
	if err != nil {
		panic(err)
	}
	
	out, _ := i.Encode(neural.Predict(gazou))
	
	fmt.Println("maybe :", out)
	
	for num, v := range neural.Predict(gazou) {
		fmt.Printf("%s : %0.1f ％\n", i.Labels[num], v*100)
	}
	sum := float64(len(ex))
	
	correct := 0.0
	
	for _, p := range ex {
		actual, _ := i.Encode(neural.Predict(p.Input))
		except, _ := i.Encode(p.Response)
		if actual == except {
			correct++
		} else {
			fmt.Printf("except:%s ,but actual:%s\n", except, actual)
		}
	}
	fmt.Printf("correct:%v, sum:%v  %0.1f％\n", correct, sum, 100*correct/sum)
}

func LoadModel(filename string) (*deep.Neural, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	model, err := deep.Unmarshal(bytes)
	if err != nil {
		return nil, err
	}
	return model, nil
}

func SaveModel(model *deep.Neural, filename string) error {
	file, err := util.OpenOrCreateFile(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	err = json.NewEncoder(file).Encode((*model).Dump())
	if err != nil {
		return err
	}
	return nil
}
```
## LICENSE
MIT
