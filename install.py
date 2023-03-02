import launch

# if not launch.is_installed("diffusers"):
#     launch.run_pip("install git+https://github.com/camenduru/diffusers@v0.9.1", "camenduru/diffusers@v0.9.1 requirements for diffusers extension")
# if not launch.is_installed("transformers"):
#     launch.run_pip("install transformers==4.23.1", "transformers==4.23.1 requirements for diffusers extension")
# if not launch.is_installed("ftfy"):
#     launch.run_pip("install ftfy==6.1.1", "ftfy==6.1.1 requirements for diffusers extension")
# if not launch.is_installed("accelerate"):
#     launch.run_pip("install accelerate==0.13.1", "accelerate==0.13.1 requirements for diffusers extension")
# if not launch.is_installed("bitsandbytes"):
#     launch.run_pip("install bitsandbytes==0.37.0", "bitsandbytes==0.37.0 requirements for diffusers extension")
# if not launch.is_installed("safetensors"):
#     launch.run_pip("install safetensors==0.2.8", "safetensors==0.2.8 requirements for diffusers extension")

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers==0.13.1", "diffusers==0.13.1 requirements for diffusers extension")
if not launch.is_installed("transformers"):
    launch.run_pip("install transformers==4.26.1", "transformers==4.26.1 requirements for diffusers extension")
if not launch.is_installed("ftfy"):
    launch.run_pip("install ftfy==6.1.1", "ftfy==6.1.1 requirements for diffusers extension")
if not launch.is_installed("accelerate"):
    launch.run_pip("install accelerate==0.16.0", "accelerate==0.16.0 requirements for diffusers extension")
if not launch.is_installed("bitsandbytes"):
    launch.run_pip("install bitsandbytes==0.37.0", "bitsandbytes==0.37.0 requirements for diffusers extension")
if not launch.is_installed("safetensors"):
    launch.run_pip("install safetensors==0.2.8", "safetensors==0.2.8 requirements for diffusers extension")