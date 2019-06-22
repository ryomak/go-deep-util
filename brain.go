package brain

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/patrikeh/go-deep/training"
	"github.com/ryomak/deep-learning-go/util"
)

/*
  MIT License

 Copyright (c) 2018 Patrik Ehrencrona

 Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
type ActivationMode int
type FuncMode int
type LossType int

type Config struct {
	ModelFile  string
	EpochNum   int
	Bias       bool
	Hiddens    []int
	Activation ActivationMode
	Mode       FuncMode
	Loss       int
}

type IBrainUtil interface {
	Decode(string) ([]float64, error)
	Encode([]float64) (interface{}, error)
	MakePattern() ([]util.DataSet, error)
}

type Brain struct {
	Util    IBrainUtil
	Config  *Config
	Model   *deep.Neural
	Trainer training.Trainer
}

func Init(config *Config, util IBrainUtil) *Brain {
	rand.Seed(0)
	return &Brain{
		Util:   util,
		Config: config,
		Model:  &deep.Neural{},
	}
}

func (b *Brain) Train(pattern []util.DataSet) error {
	c := b.Config
	log.Println("input:", len(pattern[0].Input))
	log.Println("output:", len(pattern[0].Response))
	b.Model = deep.NewNeural(&deep.Config{
		Inputs:     len(pattern[0].Input),
		Layout:     append(c.Hiddens, len(pattern[0].Response)),
		Activation: deep.ActivationType(c.Activation),
		Mode:       deep.Mode(c.Mode),
		Weight:     deep.NewNormal(1, 0),
		Bias:       c.Bias,
		Loss:       deep.LossType(c.Loss),
	})
	ex := training.Examples(util.DatsetToDataSets(pattern))
	ex.Shuffle()
	b.Trainer.Train(b.Model, ex, ex, c.EpochNum)
	return b.SaveModel()
}

func (b *Brain) Output(input []float64) []float64 {
	return b.Model.Predict(input)
}

func (b *Brain) LoadModel() error {
	cDir, err := os.Getwd()
	if err != nil {
		return err
	}
	file, err := os.Open(filepath.Join(cDir, b.Config.ModelFile))
	if err != nil {
		return err
	}
	defer file.Close()
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return err
	}
	model, err := deep.Unmarshal(bytes)
	if err != nil {
		return err
	}
	b.Model = model
	return nil
}

func (b *Brain) SaveModel() error {
	file, err := util.OpenOrCreateFile(b.Config.ModelFile)
	if err != nil {
		return err
	}
	defer file.Close()
	bm := *b.Model
	err = json.NewEncoder(file).Encode(bm.Dump())
	if err != nil {
		return err
	}
	return nil
}
