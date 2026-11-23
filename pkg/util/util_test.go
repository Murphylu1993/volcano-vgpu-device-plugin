/*
Copyright 2026 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"volcano.sh/k8s-device-plugin/pkg/config"
)

const testDeviceConfigCMName = "volcano-vgpu-device-config"

func deviceConfigCM(namespace string, data map[string]string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: testDeviceConfigCMName, Namespace: namespace},
		Data:       data,
	}
}

func TestDeviceConfigNamespaces(t *testing.T) {
	tests := []struct {
		name       string
		configured string
		want       []string
	}{
		{
			name:       "unset keeps legacy search order",
			configured: "",
			want:       []string{"kube-system", "volcano-system"},
		},
		{
			name:       "custom namespace is searched first",
			configured: "volcano",
			want:       []string{"volcano", "kube-system", "volcano-system"},
		},
		{
			name:       "legacy namespace is not searched twice",
			configured: "volcano-system",
			want:       []string{"volcano-system", "kube-system"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := deviceConfigNamespaces(tt.configured); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("deviceConfigNamespaces(%q) = %v, want %v", tt.configured, got, tt.want)
			}
		})
	}
}

func TestLoadConfigFromCM(t *testing.T) {
	tests := []struct {
		name           string
		configured     string
		objects        []runtime.Object
		wantSplitCount uint
		wantErr        bool
	}{
		{
			name:       "found in configured namespace",
			configured: "volcano",
			objects: []runtime.Object{
				deviceConfigCM("volcano", map[string]string{DeviceConfigurationConfigMapKey: "nvidia:\n  deviceSplitCount: 7\n"}),
			},
			wantSplitCount: 7,
		},
		{
			name:       "configured namespace wins over kube-system",
			configured: "volcano",
			objects: []runtime.Object{
				deviceConfigCM("volcano", map[string]string{DeviceConfigurationConfigMapKey: "nvidia:\n  deviceSplitCount: 7\n"}),
				deviceConfigCM("kube-system", map[string]string{DeviceConfigurationConfigMapKey: "nvidia:\n  deviceSplitCount: 3\n"}),
			},
			wantSplitCount: 7,
		},
		{
			name:       "falls back to kube-system",
			configured: "volcano",
			objects: []runtime.Object{
				deviceConfigCM("kube-system", map[string]string{DeviceConfigurationConfigMapKey: "nvidia:\n  deviceSplitCount: 3\n"}),
			},
			wantSplitCount: 3,
		},
		{
			name:       "falls back to volcano-system when unset",
			configured: "",
			objects: []runtime.Object{
				deviceConfigCM("volcano-system", map[string]string{DeviceConfigurationConfigMapKey: "nvidia:\n  deviceSplitCount: 5\n"}),
			},
			wantSplitCount: 5,
		},
		{
			name:       "missing everywhere",
			configured: "volcano",
			objects:    nil,
			wantErr:    true,
		},
		{
			name:       "missing key",
			configured: "volcano",
			objects: []runtime.Object{
				deviceConfigCM("volcano", map[string]string{"other.yaml": "nvidia: {}\n"}),
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs := fake.NewClientset(tt.objects...)
			got, err := loadConfigFromCM(context.Background(), cs, deviceConfigNamespaces(tt.configured), testDeviceConfigCMName)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got config %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.NvidiaConfig.DeviceSplitCount != tt.wantSplitCount {
				t.Errorf("deviceSplitCount = %d, want %d", got.NvidiaConfig.DeviceSplitCount, tt.wantSplitCount)
			}
		})
	}
}

// snapshotNvidiaGlobals returns a function that restores the global config
// values applyNvidiaConfig writes to.
func snapshotNvidiaGlobals() func() {
	schedulerConfig := config.SchedulerConfig
	splitCount, coresScaling, memoryFactor := config.DeviceSplitCount, config.DeviceCoresScaling, config.GPUMemoryFactor
	disableCoreLimit := config.DisableCoreLimit
	libCudaLogLevel := config.LibCudaLogLevel
	resourceName, resourceMem := ResourceName, ResourceMem
	resourceCores, resourceMemPercentage := ResourceCores, ResourceMemPercentage
	return func() {
		config.SchedulerConfig = schedulerConfig
		config.DeviceSplitCount = splitCount
		config.DeviceCoresScaling = coresScaling
		config.GPUMemoryFactor = memoryFactor
		config.DisableCoreLimit = disableCoreLimit
		config.LibCudaLogLevel = libCudaLogLevel
		ResourceName = resourceName
		ResourceMem = resourceMem
		ResourceCores = resourceCores
		ResourceMemPercentage = resourceMemPercentage
	}
}

// TestDisableCoreLimitSync verifies that the ConfigMap value is loaded and
// published to the global consumed by the allocate path.
func TestDisableCoreLimitSync(t *testing.T) {
	t.Cleanup(snapshotNvidiaGlobals())

	tests := []struct {
		name     string
		cmData   string
		initial  bool
		expected bool
	}{
		{
			name:     "enabled in ConfigMap",
			cmData:   "nvidia:\n  disableCoreLimit: true\n",
			initial:  false,
			expected: true,
		},
		{
			name:     "disabled in ConfigMap",
			cmData:   "nvidia:\n  disableCoreLimit: false\n",
			initial:  true,
			expected: false,
		},
		{
			name:     "defaults to false when omitted",
			cmData:   "nvidia:\n  deviceSplitCount: 3\n",
			initial:  true,
			expected: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config.DisableCoreLimit = tt.initial
			cs := fake.NewClientset(deviceConfigCM("kube-system", map[string]string{DeviceConfigurationConfigMapKey: tt.cmData}))
			got, err := loadConfigFromCM(context.Background(), cs, deviceConfigNamespaces(""), testDeviceConfigCMName)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			applyNvidiaConfig(got.NvidiaConfig)
			if config.DisableCoreLimit != tt.expected {
				t.Fatalf("DisableCoreLimit = %v, want %v", config.DisableCoreLimit, tt.expected)
			}
		})
	}
}

func intPtr(v int) *int {
	return &v
}

// TestLibCudaLogLevelSync verifies that libCudaLogLevel is synced into the
// global consumed by the allocate path, preserving an explicit 0 while keeping
// omitted/null values as unset.
func TestLibCudaLogLevelSync(t *testing.T) {
	t.Cleanup(snapshotNvidiaGlobals())

	tests := []struct {
		name   string
		cmData string
		want   *int
	}{
		{
			name:   "explicit zero is preserved",
			cmData: "nvidia:\n  libCudaLogLevel: 0\n",
			want:   intPtr(0),
		},
		{
			name:   "explicit non-zero value",
			cmData: "nvidia:\n  libCudaLogLevel: 4\n",
			want:   intPtr(4),
		},
		{
			name:   "omitted stays unset",
			cmData: "nvidia:\n  deviceSplitCount: 3\n",
			want:   nil,
		},
		{
			name:   "explicit null stays unset",
			cmData: "nvidia:\n  libCudaLogLevel: null\n",
			want:   nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Start from a non-nil value so the unset cases also prove the sync
			// clears a previously configured level.
			config.LibCudaLogLevel = intPtr(1)
			cs := fake.NewClientset(deviceConfigCM("kube-system", map[string]string{DeviceConfigurationConfigMapKey: tt.cmData}))
			got, err := loadConfigFromCM(context.Background(), cs, deviceConfigNamespaces(""), testDeviceConfigCMName)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			applyNvidiaConfig(got.NvidiaConfig)
			switch {
			case tt.want == nil && config.LibCudaLogLevel != nil:
				t.Fatalf("LibCudaLogLevel = %d, want nil (unset)", *config.LibCudaLogLevel)
			case tt.want != nil && config.LibCudaLogLevel == nil:
				t.Fatalf("LibCudaLogLevel = nil, want %d", *tt.want)
			case tt.want != nil && *config.LibCudaLogLevel != *tt.want:
				t.Fatalf("LibCudaLogLevel = %d, want %d", *config.LibCudaLogLevel, *tt.want)
			}
		})
	}
}

// TestLoadConfigFromCMStopsOnNonNotFoundError checks that an error other than
// NotFound in the configured namespace is returned instead of being hidden by a
// silent fallback to a legacy namespace.
func TestLoadConfigFromCMStopsOnNonNotFoundError(t *testing.T) {
	cs := fake.NewClientset(
		deviceConfigCM("kube-system", map[string]string{DeviceConfigurationConfigMapKey: "nvidia:\n  deviceSplitCount: 3\n"}),
	)
	forbidden := apierrors.NewForbidden(schema.GroupResource{Resource: "configmaps"}, testDeviceConfigCMName, errors.New("denied"))
	cs.PrependReactor("get", "configmaps", func(action k8stesting.Action) (bool, runtime.Object, error) {
		if action.GetNamespace() == "volcano" {
			return true, nil, forbidden
		}
		return false, nil, nil
	})

	got, err := loadConfigFromCM(context.Background(), cs, deviceConfigNamespaces("volcano"), testDeviceConfigCMName)
	if err == nil {
		t.Fatalf("expected the Forbidden error, got config %+v", got)
	}
	if !apierrors.IsForbidden(err) {
		t.Fatalf("expected a Forbidden error, got %v", err)
	}
}

func TestDecodeNodeDevicesOutOfRange(t *testing.T) {
	for _, tt := range []struct {
		name                  string
		count, memory         string
		wantCount, wantMemory int32
	}{
		{name: "positive overflow", count: "4294967297", memory: "4294967298", wantCount: math.MaxInt32, wantMemory: math.MaxInt32},
		{name: "negative overflow", count: "-4294967297", memory: "-4294967298", wantCount: math.MinInt32, wantMemory: math.MinInt32},
		{name: "count overflow", count: "2147483648", memory: "4096", wantCount: math.MaxInt32, wantMemory: 4096},
		{name: "memory overflow", count: "10", memory: "2147483648", wantCount: 10, wantMemory: math.MaxInt32},
		{name: "beyond int64", count: "9223372036854775808", memory: "-9223372036854775809", wantCount: math.MaxInt32, wantMemory: math.MinInt32},
		{name: "int32 limits", count: "2147483647", memory: "-2147483648", wantCount: math.MaxInt32, wantMemory: math.MinInt32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := DecodeNodeDevices("GPU-a," + tt.count + "," + tt.memory + ",NVIDIA,true:")
			if len(got) != 1 {
				t.Fatalf("got %d devices, want 1 with clamped values", len(got))
			}
			if got[0].Count != tt.wantCount || got[0].Devmem != tt.wantMemory {
				t.Fatalf("count=%d devmem=%d; want count=%d devmem=%d", got[0].Count, got[0].Devmem, tt.wantCount, tt.wantMemory)
			}
		})
	}
}

func pendingPod(name, node, devs string, annos map[string]string) v1.Pod {
	a := map[string]string{AssignedNodeAnnotations: node, AssignedIDsToAllocateAnnotations: devs}
	for k, v := range annos {
		a[k] = v
	}
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Annotations: a},
		Status:     v1.PodStatus{Phase: v1.PodPending},
	}
}

func TestGetOldestPod(t *testing.T) {
	const (
		small = "GPU-a,NVIDIA,683,30:;"
		big   = "GPU-a,NVIDIA,1366,30:;"
	)
	at := func(vt, bt string) map[string]string {
		return map[string]string{AssignedTimeAnnotations: vt, BindTimeAnnotations: bt}
	}
	done := pendingPod("done", "n1", small, at("1", "1"))
	done.Annotations[DeviceBindPhase] = DeviceBindSuccess
	failed := pendingPod("failed", "n1", small, at("1", "1"))
	failed.Annotations[DeviceBindPhase] = DeviceBindFailed
	running := pendingPod("running", "n1", small, at("1", "1"))
	running.Status.Phase = v1.PodRunning

	tests := []struct {
		name    string
		pods    []v1.Pod
		want    string
		wantErr bool
	}{
		{"no pods", nil, "", false},
		{"other node", []v1.Pod{pendingPod("a", "n2", small, at("1", "1"))}, "", false},
		{"device count mismatch", []v1.Pod{pendingPod("a", "n1", small+"GPU-b,NVIDIA,1,1:;", at("1", "1"))}, "", false},
		{"older vgpu-time wins", []v1.Pod{
			pendingPod("new", "n1", big, at("5", "5")),
			pendingPod("old", "n1", small, at("4", "9")),
		}, "old", false},
		{"bind-time breaks vgpu-time tie", []v1.Pod{
			pendingPod("late", "n1", big, at("5", "7")),
			pendingPod("early", "n1", small, at("5", "6")),
		}, "early", false},
		{"allocation already done is skipped", []v1.Pod{
			done, failed, running,
			pendingPod("live", "n1", big, at("9", "9")),
		}, "live", false},
		{"only finished pods", []v1.Pod{done, failed, running}, "", false},
		{"tie with different devices is ambiguous", []v1.Pod{
			pendingPod("a", "n1", small, at("5", "5")),
			pendingPod("b", "n1", big, at("5", "5")),
		}, "", true},
		{"missing timestamps tie is ambiguous", []v1.Pod{
			pendingPod("a", "n1", small, nil),
			pendingPod("b", "n1", big, nil),
		}, "", true},
		{"tie with identical devices is fine", []v1.Pod{
			pendingPod("a", "n1", small, at("5", "5")),
			pendingPod("b", "n1", small, at("5", "5")),
		}, "a", false},
		{"strictly older pod resolves an earlier tie", []v1.Pod{
			pendingPod("a", "n1", small, at("5", "5")),
			pendingPod("b", "n1", big, at("5", "5")),
			pendingPod("c", "n1", big, at("4", "4")),
		}, "c", false},
		{"tie behind an older pod is irrelevant", []v1.Pod{
			pendingPod("c", "n1", big, at("4", "4")),
			pendingPod("a", "n1", small, at("5", "5")),
			pendingPod("b", "n1", big, at("5", "5")),
		}, "c", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getOldestPod(tt.pods, "n1", 1)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			gotName := ""
			if got != nil {
				gotName = got.Name
			}
			if gotName != tt.want {
				t.Fatalf("got %q, want %q", gotName, tt.want)
			}
		})
	}
}
