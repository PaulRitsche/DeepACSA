document.addEventListener('DOMContentLoaded', function () {
  const repoOwner = 'PaulRitsche';
  const repoName = 'DeepACSA';
  const starsEl = document.getElementById('gh-stars');
  const linkEl = document.getElementById('gh-link');
  if (!starsEl) return;

  // Use GitHub public API to get repository info (stargazers_count)
  fetch(`https://api.github.com/repos/${repoOwner}/${repoName}`)
    .then((res) => {
      if (!res.ok) throw new Error('network');
      return res.json();
    })
    .then((data) => {
      const stars = data && data.stargazers_count ? data.stargazers_count : '0';
      starsEl.textContent = `${stars} ★`;
      if (linkEl) linkEl.href = `https://github.com/${repoOwner}/${repoName}`;
    })
    .catch(() => {
      // Graceful fallback
      starsEl.textContent = '★';
    });
});
