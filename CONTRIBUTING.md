# Contributing to NVIDIA cuDF plugin for Apache Spark Tools

Contributions to the cuDF plugin tools fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/NVIDIA/cudf-spark-tools/issues/new/choose)
    describing in detail the problem or new feature. The project team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/NVIDIA/cudf-spark-tools/issues/new/choose). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

## Branching Convention

There are two branches in this repository:

* `dev`: are development branches which can change often. Note that we merge into
  the branch with the greatest version number, as that is our default branch.

* `main`: is the branch with the latest released code, and the version tag (i.e. `v0.1.0`)
  is held here. `main` will change with new releases, but otherwise it should not change with
  every pull request merged, making it a more stable branch.

## Code contributions

### Code style guide

Please follow the style of the existing codebase.

* Run `pylint` over your code using `.pylintrc` file in each module.
  * line width: 120 characters
* Some pylint warnings could be incorrect. For incorrect ones, suppress those warnings by setting a line-level comment.

### Testing your changes

* Add appropriate unit-tests that covers the new changes.
* Run `tox` on the entire project. For example, to test changes committed to `user_tools`:
  ```bash
  $ cd user_tools
  $ # install tox
  $ pip install tox
  $ # run tox
  $ tox
  ```

### Sign your work

We require that all contributors sign-off on their commits. This certifies that the contribution is
your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not signed off will not be accepted.

To sign off on a commit use the `--signoff` (or `-s`) option when committing your changes:

```shell
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The sign-off is a simple line at the end of the explanation for the patch. Your signature certifies
that you wrote the patch or otherwise have the right to pass it on as an open-source patch.
Use your real name, no pseudonyms or anonymous contributions.
If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with
`git commit -s`.


The signoff means you certify the below (from [developercertificate.org](https://developercertificate.org)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```