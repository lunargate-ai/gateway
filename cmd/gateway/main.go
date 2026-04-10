package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"sync"
	"syscall"
	"time"

	"github.com/lunargate-ai/gateway/internal/api"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelselect"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/remotecontrol"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var version = "dev"

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	logLevel := flag.String("log-level", "", "Override log level (debug, info, warn, error)")
	flag.Parse()

	// --- Load Config ---
	cfgManager, err := config.NewManager(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	cfg := cfgManager.Get()
	if *logLevel != "" {
		cfg.Logging.Level = *logLevel
	}

	// --- Setup Logging ---
	effectiveLogLevel := setupLogging(cfg.Logging)
	printStartupBanner(version, *configPath, effectiveLogLevel, cfg.Server.Address())

	log.Info().
		Str("version", version).
		Str("config", *configPath).
		Str("log_level", effectiveLogLevel).
		Msg("starting LunarGate gateway")

	// --- Initialize Components ---
	registry := providers.NewRegistry(cfg.Providers)
	if len(registry.List()) == 0 {
		log.Fatal().Msg("no providers configured - at least one provider is required")
	}

	routingEngine := routing.NewEngine(cfg.Routing)
	retrier := resilience.NewRetrier(cfg.Retry)
	cbManager := resilience.NewCircuitBreakerManager()
	fallbackExec := resilience.NewFallbackExecutor(retrier, cbManager)
	cache := middleware.NewCache(cfg.Cache)
	rateLimiter := middleware.NewRateLimiter(cfg.RateLimit)
	authManager, err := security.NewManager(cfg.Security)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to initialize inbound auth")
	}
	streamer := streaming.NewHandler()
	metrics := observability.NewMetrics()
	healthChecker := health.NewChecker(version)
	collectorClient := observability.NewCollectorClient(cfg.DataSharing, version)
	selector := modelselect.NewEngine(cfg.ModelSelect)
	store := modelstore.NewStore(registry, cfg.Providers)
	handler := api.NewHandler(registry, routingEngine, fallbackExec, cache, streamer, metrics, collectorClient, selector, store)
	handler.UpdateProviderConfigs(cfg.Providers)
	remoteControlBaseURL := "http://" + localLoopbackAddress(cfg.Server)
	modelIDs := func(ctx context.Context) []string {
		var ids []string
		if store != nil {
			for _, item := range store.AllModels(ctx) {
				if item.ID != "" {
					ids = append(ids, item.ID)
				}
			}
		}
		return ids
	}
	var remoteControlMu sync.Mutex
	var remoteControlClient *remotecontrol.Client
	remoteControlCancel := func() {}
	refreshRemoteControlHello := func() {
		remoteControlMu.Lock()
		defer remoteControlMu.Unlock()
		if remoteControlClient != nil {
			remoteControlClient.RefreshHello()
		}
	}
	reconcileRemoteControl := func(cfg *config.Config) {
		remoteControlMu.Lock()
		defer remoteControlMu.Unlock()

		remoteControlCancel()
		remoteControlCancel = func() {}
		remoteControlClient = remotecontrol.NewClient(
			cfg.DataSharing,
			version,
			remoteControlBaseURL,
			routingEngine.RouteNames,
			modelIDs,
		)
		if remoteControlClient != nil {
			rcCtx, cancel := context.WithCancel(context.Background())
			remoteControlCancel = cancel
			remoteControlClient.Start(rcCtx)
		}

		logRemoteControlStatus(cfg, remoteControlClient)
	}
	currentCfg := cfg
	reconcileRemoteControl(cfg)

	// --- Setup Hot-Reload ---
	cfgManager.OnChange(func(newCfg *config.Config) {
		oldCfg := currentCfg
		currentCfg = newCfg
		providersChanged := !reflect.DeepEqual(oldCfg.Providers, newCfg.Providers)
		routingChanged := !reflect.DeepEqual(oldCfg.Routing, newCfg.Routing)
		modelSelectChanged := !reflect.DeepEqual(oldCfg.ModelSelect, newCfg.ModelSelect)
		dataSharingChanged := !reflect.DeepEqual(oldCfg.DataSharing, newCfg.DataSharing)
		securityChanged := !reflect.DeepEqual(oldCfg.Security, newCfg.Security)
		serverChanged := !reflect.DeepEqual(oldCfg.Server, newCfg.Server)

		if serverChanged {
			log.Warn().Msg("server config changed; listen address and timeouts still require process restart to fully apply")
		}

		setupLogging(newCfg.Logging)
		routingEngine.UpdateConfig(newCfg.Routing)
		if registry.UpdateProvidersConfig(newCfg.Providers) {
			store.UpdateProvidersConfig(newCfg.Providers)
			handler.UpdateProviderConfigs(newCfg.Providers)
		}
		rateLimiter.UpdateConfig(newCfg.RateLimit)
		cache.UpdateConfig(newCfg.Cache)
		retrier.UpdateConfig(newCfg.Retry)
		selector.UpdateConfig(newCfg.ModelSelect)
		collectorClient.UpdateConfig(newCfg.DataSharing)
		if securityChanged {
			if err := authManager.UpdateConfig(newCfg.Security); err != nil {
				log.Error().Err(err).Msg("failed to reconcile inbound auth config; keeping previous auth state")
			}
		}

		if dataSharingChanged {
			reconcileRemoteControl(newCfg)
		} else if providersChanged || routingChanged || modelSelectChanged {
			refreshRemoteControlHello()
			remoteControlMu.Lock()
			clientSnapshot := remoteControlClient
			remoteControlMu.Unlock()
			logRemoteControlStatus(newCfg, clientSnapshot)
		}
		log.Info().Msg("hot-reload: routing, providers, retry, cache, rate limit, inbound auth, model selection, collector, and remote control reconciled")
	})
	cfgManager.WatchChanges()

	// --- Create API Handler & Router ---
	router := api.NewRouter(handler, authManager, rateLimiter, healthChecker)

	// --- Start HTTP Server ---
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.Server.ReadTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
		IdleTimeout:       cfg.Server.IdleTimeout,
		MaxHeaderBytes:    1 << 20,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Info().
			Str("address", cfg.Server.Address()).
			Strs("providers", registry.List()).
			Msg("gateway listening")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("server failed")
		}
	}()

	<-done
	log.Info().Msg("shutting down gateway...")

	healthChecker.SetReady(false)
	remoteControlMu.Lock()
	remoteControlCancel()
	remoteControlMu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error().Err(err).Msg("server shutdown error")
	}
	if collectorClient != nil {
		collectorClient.Stop()
	}
	if cache != nil {
		cache.Stop()
	}

	log.Info().Msg("gateway stopped")
}

func setupLogging(cfg config.LoggingConfig) string {
	level, err := zerolog.ParseLevel(cfg.Level)
	if err != nil {
		level = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(level)

	if cfg.Format == "console" {
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339})
	}

	zerolog.TimeFieldFormat = time.RFC3339Nano

	return level.String()
}

func printStartupBanner(version, configPath, logLevel, address string) {
	fmt.Fprintf(os.Stderr, `
 _                           ____       _
| |   _   _ _ __   __ _ _ _ / ___| __ _| |_ ___
| |  | | | | '_ \ / _`+"`"+` | '_| |  _ / _`+"`"+` | __/ _ \
| |__| |_| | | | | (_| | | | |_| | (_| | ||  __/
|_____\__,_|_| |_|\__,_|_|  \____|\__,_|\__\___|

version: %s
config:  %s
log:     %s
listen:  %s

`, version, configPath, logLevel, address)
}

func localLoopbackAddress(cfg config.ServerConfig) string {
	host := cfg.Host
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return fmt.Sprintf("%s:%d", host, cfg.Port)
}

func logRemoteControlStatus(cfg *config.Config, remoteControlClient *remotecontrol.Client) {
	remoteControlEnabled := cfg.DataSharing.RemoteControl && remoteControlClient != nil
	remoteControlInstanceID := ""
	if remoteControlClient != nil {
		remoteControlInstanceID = remoteControlClient.InstanceID()
	}

	log.Info().
		Bool("data_sharing_enabled", cfg.DataSharing.Enabled).
		Bool("remote_control_enabled", remoteControlEnabled).
		Str("instance_id", remoteControlInstanceID).
		Str("backend_url", cfg.DataSharing.BackendURL).
		Msg("gateway data sharing and remote control status")
}
