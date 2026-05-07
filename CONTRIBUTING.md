# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests

> **Prerequisite**: This repository uses [Git LFS](https://git-lfs.github.com/) for large binary files (images, archives). Before cloning, install Git LFS:
> ```bash
> # macOS: brew install git-lfs
> # Ubuntu/Debian: sudo apt-get install git-lfs
> git lfs install
> ```

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

### Blog content

Blog posts live on the `content` branch and are AWS-authored. PRs from non-maintainers targeting `content` are auto-closed. If you have feedback or topic suggestions, please open an issue.


## Contributions format
Assets in this repository are organized differently depending on their nature but regardless of that each test example, architecture must have the following:

- Must be self contained with documentation and scripts. Dependencies are accepted for the network architecture (VPC templates) and if they are within the repository. If your scripts depend on external software (libs, frameworks, containers...) then fix the versions via a tag or commit ID to ensure reproducibility and do not use a `latest` tag.
- A README file in markdown format with the prerequisites, instructions on deploying the architecture or running the test case and notes for known edge cases or common issues.
- Scripts templates must be numbered in sequence in their usage order starting by 0. Example: `0.preprocessing.sh`, `1.processing.sh`, `2.post-processing.sh`.
- All contributions must be done through PRs at the exception of hotfixes.
- Contributions should follow a KISS principle, **K**eep **I**t **S**imple and **S**tupid. Users should expect to copy paste commands, reproduce tests as is and have some guidance on what to modify if they need to.

For guidance specific on assets:

- *Architectures*: self-contained, ensure you stick to AWS CloudFormation, the AWS CDK or Terraform.
- *Machine Image*: based on Packer and use Ansible. No specific guidance.
- *Test cases*: self-contained models & frameworks test cases, please ensure that these are tested and run at the target scale (if you say 256 A100, test on that scale).
- *Validation scripts*: self contained, keep them simple enough, no need to boil the ocean.

## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
