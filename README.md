# go-deep-assets
[![GoReport](https://goreportcard.com/badge/github.com/ryomak/go-deep-assets)](https://goreportcard.com/report/github.com/ryomak/go-deep-assets)  

go-deep-assets is assets for [github.com/patrikeh/go-deep](https://github.com/patrikeh/go-deep)

## install
```
$ go get https://github.com/ryomak/go-deep-assets
```
## example
### directory

```
.
├── dataset
│   ├── lulu
│   ├── tida
│   └── yuna
├── input.jpg
└── main.go
```
lulu/tida/yuna folder -> *.png/*.jpg

```go

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/briandowns/spinner"
	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	assets "github.com/ryomak/go-deep-assets"
	iclass "github.com/ryomak/go-deep-assets/iclassifier"
	util "github.com/ryomak/go-deep-assets/util"
)

func main() {

	loading := spinner.New(spinner.CharSets[9], 100*time.Millisecond)

	labels := []string{
		"tida",
		"yuna",
		"lulu",
	}

	var i assets.IBrainUtil

	i = iclass.Init(
		labels,
		"dataset",
		30,
		30,
	)
	rand.Seed(time.Now().UnixNano())

	data, err := i.MakePattern()
	if err != nil {
		panic(err)
	}

	//shuffle
	ex := training.Examples(util.DatsetToDataSets(data))
	ex.Shuffle()

	neural := deep.NewNeural(&deep.Config{
		Inputs:     len(data[0].Input),
		Layout:     append([]int{1000, 100}, len(data[0].Response)),
		Activation: deep.ActivationSoftmax, Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(1, 0),
		Bias:       true,
	})

	fmt.Printf("train start[testcase:%d] ...\n", len(data))
	loading.Start()
	trainer := training.NewBatchTrainer(training.NewAdam(0.001, 0, 0, 0), 40, len(ex)/2, 12)
	training, heldout := ex.Split(0.8)
	trainer.Train(neural, training, heldout, 100)
	loading.Stop()

	inputFile := "input.jpg"
	gazou, _ := i.Decode(inputFile)
	out, _ := i.Encode(neural.Predict(gazou))

	fmt.Printf("[Result]\n%s is maybe : %v \n", inputFile, out)

	doTest(neural, ex, i)
}

func doTest(neural *deep.Neural, ex training.Examples, i assets.IBrainUtil) {

	fmt.Println("Test start with learned Model")
	sum := float64(len(ex))
	correct := 0.0

	for _, p := range ex {
		actual, _ := i.Encode(neural.Predict(p.Input))
		except, _ := i.Encode(p.Response)
		if actual == except {
			correct++
		} else {
			fmt.Printf("miss:except: %s,but actual: %s\n", except, actual)
		}
	}
	fmt.Printf("[Test Result]\ncorrect:%v, sum:%v  %0.1f％\n", correct, sum, 100*correct/sum)
}

```
## LICENSE
MIT
