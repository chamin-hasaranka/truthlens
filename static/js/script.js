/* ==========================================================================
   TruthLens AI — Main JavaScript
   Handles: dark/light theme toggle (persisted in-memory), scroll-triggered
   animations, password show/hide toggles, and small UX enhancements.
   ========================================================================== */

(function () {
    "use strict";

    // ---------------------------------------------------------------------
    // Theme Toggle (Dark / Light)
    // Note: We intentionally avoid localStorage/sessionStorage since this
    // app may be embedded in environments where browser storage is
    // restricted. Theme preference simply defaults to dark on each load.
    // ---------------------------------------------------------------------
    const root = document.documentElement;
    const themeToggleBtn = document.getElementById("themeToggle");
    const themeIcon = document.getElementById("themeIcon");

    function applyTheme(theme) {
        root.setAttribute("data-bs-theme", theme);
        if (themeIcon) {
            themeIcon.className = theme === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
        }
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", function () {
            const current = root.getAttribute("data-bs-theme") || "dark";
            const next = current === "dark" ? "light" : "dark";
            applyTheme(next);
        });
    }

    // ---------------------------------------------------------------------
    // Scroll-triggered fade-up animations
    // ---------------------------------------------------------------------
    function initScrollAnimations() {
        const animatedEls = document.querySelectorAll("[data-animate]");
        if (!animatedEls.length) return;

        const observer = new IntersectionObserver(
            function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        const delay = entry.target.getAttribute("data-delay") || 0;
                        setTimeout(function () {
                            entry.target.classList.add("in-view");
                        }, parseInt(delay, 10));
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.12 }
        );

        animatedEls.forEach(function (el) {
            observer.observe(el);
        });
    }

    // ---------------------------------------------------------------------
    // Password visibility toggle (used on login/register pages)
    // ---------------------------------------------------------------------
    function initPasswordToggles() {
        document.querySelectorAll(".toggle-password").forEach(function (toggle) {
            toggle.addEventListener("click", function () {
                const targetId = toggle.getAttribute("data-target");
                const input = document.getElementById(targetId);
                if (!input) return;

                const icon = toggle.querySelector("i");
                if (input.type === "password") {
                    input.type = "text";
                    if (icon) icon.className = "bi bi-eye-slash";
                } else {
                    input.type = "password";
                    if (icon) icon.className = "bi bi-eye";
                }
            });
        });
    }

    // ---------------------------------------------------------------------
    // Auto-dismiss flash messages after a few seconds
    // ---------------------------------------------------------------------
    function initAutoDismissAlerts() {
        const alerts = document.querySelectorAll(".alert");
        alerts.forEach(function (alert) {
            setTimeout(function () {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                if (bsAlert) bsAlert.close();
            }, 6000);
        });
    }

    // ---------------------------------------------------------------------
    // Init on DOM ready
    // ---------------------------------------------------------------------
    document.addEventListener("DOMContentLoaded", function () {
        initScrollAnimations();
        initPasswordToggles();
        initAutoDismissAlerts();
    });
})();
