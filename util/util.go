package util

import (
	"os"
	"path/filepath"

	"github.com/patrikeh/go-deep/training"
)

type DataSet training.Example
type DataSets training.Examples

func OpenOrCreateFile(path string) (*os.File, error) {
	cDir, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	file, err := os.OpenFile(filepath.Join(cDir, path), os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}
	return file, nil
}

func OpenDirFiles(path string) ([]string, error) {
	cDir, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	file, err := os.Open(filepath.Join(cDir, path))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	names, err := file.Readdirnames(-1)
	if err != nil {
		return nil, err
	}
	return names, nil
}

func DatsetToDataSets(p []DataSet) DataSets {
	res := make(DataSets, len(p))
	for i, v := range p {
		res[i] = training.Example(v)
	}
	return res
}
