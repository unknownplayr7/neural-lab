# Security policy

Neural Lab is a local macOS application. It does not require API keys, credentials, or a hosted Neural Lab account.

## Reporting a concern

Please use GitHub's private vulnerability reporting feature if it is enabled for this repository. Otherwise, contact the repository owner privately before publishing details that could put users at risk.

## Security controls and limitations

- Checkpoints are loaded with PyTorch's `weights_only=True` mode. The application intentionally does not fall back to unrestricted pickle loading on older PyTorch versions.
- Uploaded checkpoints are limited to 25 MiB and must declare the supported CNN architecture and valid class metadata.
- Uploaded images are limited to 10 MiB and 20 megapixels. Image content is decoded and normalized locally.
- Checkpoint saves are restricted to a filename inside the local `checkpoints/` directory.
- Streamlit is intended to bind to localhost. Exposing it to a LAN or the public internet is outside the supported threat model.
- Hugging Face datasets and package dependencies are external inputs. Keep dependencies updated and review security advisories.
- This is an educational application, not a hardened multi-user service. Do not expose it to untrusted users or process sensitive images.

## Safe use

- Do not commit tokens, credentials, private datasets, or personal images.
- Only load checkpoints from sources you trust, even with restricted weights-only deserialization.
- Do not upload images you do not have permission to use.
