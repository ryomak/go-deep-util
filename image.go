package assets

import (
	"github.com/ryomak/go-deep-assets/util"
)

// IBrainUtil is making data assets interface
type IBrainUtil interface {
	Decode(string) ([]float64, error)
	Encode([]float64) (interface{}, error)
	MakePattern() ([]util.DataSet, error)
}
