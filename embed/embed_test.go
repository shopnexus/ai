package embed

import (
	"errors"
	"math"
	"testing"
)

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	normalize(v)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Errorf("v = %v", v)
	}
	zero := []float32{0, 0}
	normalize(zero) // must not divide by zero
	if zero[0] != 0 || zero[1] != 0 {
		t.Errorf("zero = %v", zero)
	}
}

func TestInputErrorUnwraps(t *testing.T) {
	err := error(&InputError{Index: 2, Err: ErrBadImage})
	if !errors.Is(err, ErrBadImage) {
		t.Error("InputError does not unwrap to ErrBadImage")
	}
	if got := err.Error(); got != "input 2: bad image" {
		t.Errorf("Error() = %q", got)
	}
}

func TestNewRefusesAMissingFolder(t *testing.T) {
	_, err := New(Config{Models: map[string]string{"x": t.TempDir()}, NumThreads: 1, MaxConcurrent: 1})
	if err == nil {
		t.Fatal("New loaded a folder with no model")
	}
}
