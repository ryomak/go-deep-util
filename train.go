package brain

import (
	"github.com/patrikeh/go-deep/training"
)

type TrainType int

func (b *Brain) NewAdamTrainer(lf float64, dataSetNum int) {
	b.Trainer = training.NewBatchTrainer(training.NewAdam(lf, 0.0, 0.0, 0.0), 50, dataSetNum, 5)
}

func (b *Brain) NewSGDTrainer(lf, decay, momentum float64, dataSetNum, verbosity int, nesterov bool) {
	b.Trainer = training.NewBatchTrainer(training.NewSGD(lf, momentum, decay, nesterov), verbosity, dataSetNum, 5)
}
