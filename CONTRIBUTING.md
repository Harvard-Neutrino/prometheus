# How to contribute to Prometheus

Welcome and thank you for investing your time in contributing to Prometheus ❤️

The types of contributions we would love to accept are:

- bug reports, enhancement suggestions
- code, documentation devops improvements
- questions and discussions

## Bug reports and enhancement suggestions

If you found a bug, or came up with an idea of how we can improve our software, you can tell us about it by opening an issue:

1. Navigate to the [issue template chooser](https://github.com/Harvard-Neutrino/prometheus/issues/new/choose).
2. Select a form with a relevant issue type.
3. Fill in the form.

Not sure which form to choose or if your suggestion warrants creating an issue? You can always ask a question in the [Discussions section](https://github.com/Harvard-Neutrino/prometheus/discussions) instead.

**Best practices**:

1. When creating issues, include **as much detail as you can**: your OS, config, software versions, log outputs, anything else that can help us quickly get up to speed.
2. Before creating an issue, have a look through [the existing issues list](https://github.com/Harvard-Neutrino/prometheus/issues) and see if whatever you are trying to report, has already been reported. If that's the case and the issue is blocking your work, you can always let us know by creating [a dicussion](https://github.com/Harvard-Neutrino/prometheus/discussions).

## Code changes

This is our most preferred contribution type, since we are a small team with very limited resources, and Prometheus still needs a lot of love. So if you would like to improve our code, tests, docs, scripts, setup, or anything else, please open a pull request and we will be thrilled to review it!

To do so, you can follow a [standard open source contribution flow](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork):

- fork Prometheus repository
- create a branch, make your changes
- open a pull request, request our review

>[!TIP]
> If you are new to GitHub flows, forks, pull requests and such, you can learn more about it from [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests).

### Best practices

#### Enable maintainer edits

Sometimes the asynchronous collaboration via emails and pull request comments can take excessive amounts of time, especially if it's in regartds to small issues not directly related to the change you're proposing (versioning, compilation, etc.). To allow us to fix such issues qickly, we recommend you to [enable maintainer edits](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork) to your pull request.

You don't have to though, it's your choice.

#### Include context

In your PR description, reference any related issues, discussions, resources, add as much detail as you can.

Screenshots, demos, video walkthroughs of your changes are generally very helpful for building context, we suggest you consider including those.

#### Use good commit messages

Your commit history can be very helpful of making sense of your change or tracing issues after it is merged to the main branch of Prometheus.

Some resources on good commit messages:
- [General advice on commit message practices from Tim Pope](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
- [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/)

It's a small and relaxed project, so you don't have to follow the conventions described above completely, but we'll appreciate it if you at least use short and descriptive commit messages that read like a recipe.

Good examples:

- `add setup setup script for installation from source`
- `update docs to include contribution guidelines`

Bad examples:

- `fixed a bug`
- `amend readme`

>[!TIP]
>If you are not used to this type of workflow and are completely overwhelmed by this ask, remember that you can always [rewrite your commit history](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) with `git rebase -i` command before submitting your pull request to us.

#### Use the checklist in the pull request template

Pull request template includes a handy checklist:

```md
[ ] Code compiles without errors & works
[ ] Code changes include relevant comments & unit tests
[ ] All unit tests pass
[ ] Documentation is updated
```

You don't have to follow it to a T, but it't there to make sure your proposed change adheres to good coding practices: doesn't break anything, has unit tests, docs etc.

## Documentation changes

We love the documentation edits that fix typos, correct grammar, punctuation, facts, or expand the explanation of existing features, when the expansion has a compelling reason.

We choose not to accept edits made purely for tone or readability, unless there is a good reason for them, like many open issues or discussions with people being visibly confused by existing documentation.

The existing docs in this project are written using [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/).

For documentation best practices, refer to [Best practices for GitHub Docs](https://docs.github.com/en/contributing/writing-for-github-docs/best-practices-for-github-docs).

## Questions

If you have a question on how Prometheus works, or need some help setting up and using it, you can ask us a question in [discussions](https://github.com/Harvard-Neutrino/prometheus/discussions).

Our response may not be super prompt, but we are notified about all new discussions, and try to get to them as soon as we can.

If you are completely blocked and we cannot get to your question in time, please feel free to contact the [authors of this software](./README.md#authors) using the email addresses listed on their Inspire-HEP pages.

## A note for first time contributors

If you are thinking of contributing to Prometheus, but you have never contributed to open-source projects before, here's a few tips:

- **Good first issue**: the issues we consider beginner-friendly are labeled with a `good-first-issue` label. You can filter the issues by this label [in the issues view](https://github.com/Harvard-Neutrino/prometheus/issues?q=state%3Aopen%20label%3A%22good%20first%20issue%22).
- **Use the software**: If you are completely confused and don't know where to start, the best thing to do is to "play" with the software. Set it up, use it, and the improvement needs will quickly become clear.
- **Ask questions** Contributing to a completely new project can be intimidating at first, but we are here to help! Feel free to ask us anything by [opening a discussion](https://github.com/Harvard-Neutrino/prometheus/discussions).

___

Happy contributing!
