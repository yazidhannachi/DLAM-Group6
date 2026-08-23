import tempfile
from git import Repo, InvalidGitRepositoryError

def get_git_info(repo_path="."):
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        branch = repo.active_branch.name if not repo.head.is_detached else "DETACHED"
        is_dirty = repo.is_dirty(untracked_files=True)
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": is_dirty,
            "repo_root": repo.working_tree_dir,
        }, repo
    except InvalidGitRepositoryError:
        return {
            "git_commit": "N/A",
            "git_branch": "N/A",
            "git_dirty": "N/A",
            "repo_root": "N/A",
        }, None

def write_git_diff(repo):
    if repo is None:
        return None

    diff = repo.git.diff()
    if not diff:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".patch", mode="w")
    tmp.write(diff)
    tmp.close()
    return tmp.name