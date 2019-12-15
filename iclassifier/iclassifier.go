package iclassifier

import (
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/muesli/smartcrop"
	"github.com/muesli/smartcrop/nfnt"
	"github.com/nfnt/resize"
	"github.com/ryomak/go-deep-util"
)

type ImageClassifierUtil struct {
	LearnDir    string
	Labels      []string
	ImageHeight int
	ImageWidth  int
}

func Init(labels []string, learnDir string, imageHeight, imageWidth int) ImageClassifierUtil {
	return ImageClassifierUtil{
		LearnDir:    learnDir,
		Labels:      labels,
		ImageHeight: imageHeight,
		ImageWidth:  imageWidth,
	}
}

func (i ImageClassifierUtil) Decode(path string) ([]float64, error) {
	cDir, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	file, err := os.Open(filepath.Join(cDir, path))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	src, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	src = i.cropping(src)
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w < h {
		w = h
	} else {
		h = w
	}
	binData := make([]float64, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x, y).RGBA()
			binData[y*w*3+x*3] = float64(r>>8) / 255.0
			binData[y*w*3+x*3+1] = float64(g>>8) / 255.0
			binData[y*w*3+x*3+2] = float64(b>>8) / 255.0
		}
	}
	return binData, nil
}

func (i ImageClassifierUtil) Encode(b []float64) (interface{}, error) {
	labelNum := i.floatToLabelNum(b)
	for num, label := range i.Labels {
		if labelNum == num {
			return label, nil
		}
	}
	return labelNum, errors.New("did not match")
}

func (i ImageClassifierUtil) MakePattern() ([]util.DataSet, error) {
	patterns := []util.DataSet{}
	for n, label := range i.Labels {
		learnSet, err := i.loadDataSet(filepath.Join(i.LearnDir, label))
		if err != nil {
			return nil, err
		}
		for _, learn := range learnSet {
			patterns = append(patterns, util.DataSet{Input: learn, Response: i.labelNumToFloat(n)})
		}
	}
	return patterns, nil
}

func (i ImageClassifierUtil) loadDataSet(dirPath string) ([][]float64, error) {
	data := [][]float64{}
	names, err := util.OpenDirFiles(dirPath)
	if err != nil {
		return nil, err
	}
	for _, filename := range names {
		fname := filepath.Join(dirPath, filename)
		ff, err := i.Decode(fname)
		if err != nil {
			log.Println(fname, " can't decode")
			continue
		}
		data = append(data, ff)
	}
	return data, nil
}

func (i ImageClassifierUtil) cropping(img image.Image) image.Image {
	analyzer := smartcrop.NewAnalyzer(nfnt.NewDefaultResizer())
	topCrop, err := analyzer.FindBestCrop(img, i.ImageHeight, i.ImageWidth)
	if err == nil {
		type SubImager interface {
			SubImage(r image.Rectangle) image.Image
		}
		img = img.(SubImager).SubImage(topCrop)
	}
	return resize.Resize(uint(i.ImageHeight), uint(i.ImageWidth), img, resize.Lanczos3)
}

func (i ImageClassifierUtil) labelNumToFloat(num int) []float64 {
	f := make([]float64, len(i.Labels))
	f[num] = 1
	return f[:]
}

func (i ImageClassifierUtil) floatToLabelNum(d []float64) int {
	maxIndex := 0
	max := 0.0
	for i, v := range d {
		if max < v {
			max = v
			maxIndex = i
		}
	}
	return maxIndex
}
